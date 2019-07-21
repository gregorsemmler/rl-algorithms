# -*- coding: utf-8 -*-
import collections
from enum import Enum

import gym
import numpy as np
import re
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult


logger = logging.getLogger(__file__)


class MCAlgorithm(Enum):
    MC_FIRST_VISIT_PREDICTION = "MONTE_CARLO_FIRST_VISIT_PREDICTION"
    MC_EVERY_VISIT_PREDICTION = "MONTE_CARLO_EVERY_VISIT_PREDICTION"


class MonteCarloAgent(object):

    def __init__(self):
        self.v_table = {}

    def monte_carlo_prediction(self, env, policy, first_visit=True, gamma=0.99, num_iterations=1000):
        self.v_table = {}
        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)
            returns = collections.defaultdict(list)
            first_visited = {}

            while not done:
                action = policy(state)
                new_state, reward, done, _ = env.step(action)
                episode_result.append(action, reward, new_state)

            g = 0
            if first_visit:
                for i, s in enumerate(episode_result.states):
                    if s not in first_visited:
                        first_visited[s] = i

            i = len(episode_result.states) - 2
            while i >= 0:
                g = gamma * g + episode_result.rewards[i]
                state = episode_result.states[i]
                if not first_visit or state in first_visited and first_visited[state] == i:
                    returns[state].append(g)
                    self.v_table[state] = sum(returns[state]) / len(returns[state])

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        return self.v_table

    def predict(self, env, policy, algorithm, gamma=0.99):
        if algorithm == MCAlgorithm.MC_FIRST_VISIT_PREDICTION:
            self.monte_carlo_prediction(env, policy, first_visit=True, gamma=gamma, num_iterations)
        elif algorithm == MCAlgorithm.MC_EVERY_VISIT_PREDICTION:
            self.monte_carlo_prediction(env, policy, first_visit=True, gamma=gamma)
        raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def main():
    env_name = "Taxi-v2"
    algorithm = MCAlgorithm.MC_FIRST_VISIT_PREDICTION
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 10000
    agent = MonteCarloAgent()
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_random_steps = 1000
    num_test_episodes = 100
    while True:
        agent.learn(environment, algorithm, gamma=gamma, num_random_steps=num_random_steps)
        round_test_returns, round_test_best_result, round_test_best_return = agent.play(test_env, gamma=gamma,
                                                                                        num_episodes=num_test_episodes)
        for r_idx, r in enumerate(round_test_returns):
            writer.add_scalar("test_return", r, len(test_returns) + r_idx)

        test_returns.extend(round_test_returns)

        if test_best_return < round_test_best_return:
            test_best_return = round_test_best_return
            test_best_result = round_test_best_result

        average_test_return = 0.0 if len(round_test_returns) == 0 else sum(round_test_returns) / len(round_test_returns)
        logger.warning("Average returns: {}".format(average_test_return))

        if (goal_returns is not None and average_test_return >= goal_returns) or k >= max_rounds:
            logger.warning("Done in {} rounds!".format(k))
            break
        k += 1

    print("")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    policy = TabularPolicy()
    print("pass")