import collections
from enum import Enum

import gym
import numpy as np
import re
from math import inf
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonGreedyTabularPolicy, StateActionValueTable, CustomPolicy

logger = logging.getLogger(__file__)


class TDAlgorithm(Enum):
    TD0_PREDICTION = "TD0_PREDICTION"
    Q_LEARNING = "Q_LEARNING"
    EPSILON_GREEDY_SARSA = "EPSILON_GREEDY_SARSA"
    EPSILON_GREEDY_EXPECTED_SARSA = "EPSILON_GREEDY_EXPECTED_SARSA"
    DOUBLE_Q_LEARNING = "DOUBLE_Q_LEARNING"


class TDAgent(object):

    def __init__(self):
        self.v_table = {}
        self.policy = None
        self.q_table = None
        self.q_table2 = None

    def play(self, env, num_episodes=100, gamma=0.99, render=False):
        i = 0
        best_return = float("-inf")
        best_result = None
        episode_returns = []
        while i < num_episodes:
            state = env.reset()
            state = str(state)
            done = False

            episode_result = EpisodeResult(env, state)
            while not done:
                if render:
                    env.render()

                action = self.policy(state)
                new_state, reward, done, info = env.step(action)
                new_state = str(new_state)

                episode_result.append(action, reward, new_state)

                state = new_state

            episode_return = episode_result.calculate_return(gamma)
            if best_return < episode_return:
                best_return = episode_return
                best_result = episode_result
                logger.info("New best return: {}".format(best_return))

            episode_returns.append(episode_return)
            i += 1

        return episode_returns, best_result, best_return

    def tabular_td0(self, env, policy, num_iterations=1000, gamma=0.99, alpha=0.5):
        self.v_table = collections.defaultdict(float)
        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = policy(state)
                new_state, reward, done, _ = env.step(action)
                new_state = str(new_state)
                episode_result.append(action, reward, new_state)

                update = reward + gamma * self.v_table[new_state] - self.v_table[state]
                update *= alpha
                self.v_table[state] += update
                state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99, alpha=0.5):
        if algorithm == TDAlgorithm.TD0_PREDICTION:
            self.tabular_td0(env, policy, num_iterations=num_iterations, gamma=gamma, alpha=alpha)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))

    def learn(self, env, policy, algorithm, num_iterations, gamma=0.99, alpha=0.5, num_random_steps=1000, v=None,
              b=None):
        if algorithm == DPAlgorithm.POLICY_ITERATION:
            self.policy_iteration(env, policy, b=b, v=v, gamma=gamma, theta=alpha,
                                  num_exploration_steps=num_random_steps, max_iterations=num_iterations)
        elif algorithm == DPAlgorithm.VALUE_ITERATION:
            self.value_iteration(env, policy, b=b, v=v, gamma=gamma)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def prediction():
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_name = "FrozenLake-v0"
    algorithm = TDAlgorithm.TD0_PREDICTION
    environment = gym.make(env_name)

    gamma = 0.99
    agent = TDAgent()
    alpha = 0.5
    num_iterations = 10000
    agent.predict(environment, policy, algorithm, alpha=alpha, gamma=gamma, num_iterations=num_iterations)

    print("")
    pass


def control():
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "FrozenLake-v0"
    algorithm = DPAlgorithm.POLICY_ITERATION
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 1000
    agent = DPAgent()
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 1000
    num_test_episodes = 100
    while True:
        agent.predict(environment, policy, algorithm, b=policy, gamma=gamma, num_iterations=num_iterations)
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
    pass


if __name__ == "__main__":
    prediction()
