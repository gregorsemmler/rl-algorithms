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
            first_visited = {state: 0}

            while not done:
                action = policy(state)
                state, reward, done, _ = env.step(action)
                state = str(state)
                episode_result.append(action, reward, state)
                if first_visit:
                    if state not in first_visited:
                        first_visited[state] = len(episode_result.rewards)

            g = 0

            j = len(episode_result.states) - 2
            while j >= 0:
                g = gamma * g + episode_result.rewards[j]
                state = episode_result.states[j]
                if not first_visit or state in first_visited and first_visited[state] == j:
                    returns[state].append(g)
                    self.v_table[state] = sum(returns[state]) / len(returns[state])
                j -= 1

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99):
        if algorithm == MCAlgorithm.MC_FIRST_VISIT_PREDICTION:
            self.monte_carlo_prediction(env, policy, first_visit=True, gamma=gamma, num_iterations=num_iterations)
        elif algorithm == MCAlgorithm.MC_EVERY_VISIT_PREDICTION:
            self.monte_carlo_prediction(env, policy, first_visit=False, gamma=gamma, num_iterations=num_iterations)
        raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def main():
    env_name = "Taxi-v2"
    algorithm = MCAlgorithm.MC_FIRST_VISIT_PREDICTION
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    gamma = 0.99

    # writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    agent = MonteCarloAgent()
    for i in range(100):
        policy = TabularPolicy(random_defaults=environment.action_space.n)
        agent.predict(environment, policy, MCAlgorithm.MC_EVERY_VISIT_PREDICTION, num_iterations=1)
    print("")


if __name__ == "__main__":
    main()
    # policy = TabularPolicy(random_defaults=2)
    # x = policy("a")
    # y = policy("b")
    # z = policy("c")
    # print("pass")