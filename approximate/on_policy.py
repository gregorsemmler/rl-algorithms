import collections
from enum import Enum

import torch
from torch import optim, nn
import gym
import numpy as np
import re
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonGreedyTabularPolicy, StateActionValueTable, CustomPolicy, \
    ApproximateValueFunction

logger = logging.getLogger(__file__)


class OnPolicyAlgorithm(Enum):
    GRADIENT_MONTE_CARLO_PREDICTION = "GRADIENT_MONTE_CARLO_PREDICTION"
    SEMI_GRADIENT_TD_0 = "SEMI_GRADIENT_TD_0"


class ApproximateAgent(object):

    def __init__(self):
        self.v = None
        pass

    def gradient_monte_carlo_prediction(self, env, policy, num_iterations=10000, batch_size=32, gamma=0.99, alpha=0.1,
                                        summary_writer: SummaryWriter = None, summary_prefix=""):
        self.v = ApproximateValueFunction(env.observation_space.n, alpha)
        i = 0
        total_losses = []

        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = policy(state)
                state, reward, done, _ = env.step(action)
                state = str(state)
                episode_result.append(action, reward, state)

            g = 0

            j = len(episode_result.states) - 2
            while j >= 0:
                g = gamma * g + episode_result.rewards[j]
                state = episode_result.states[j]

                self.v.append_x_y_pair(state, g)
                j -= 1

            losses = self.v.approximate(batch_size)
            if summary_writer is not None:
                for l_idx, l in enumerate(losses):
                    summary_writer.add_scalar(f"{summary_prefix}batch_loss", l, len(total_losses) + l_idx)

            total_losses.extend(losses)

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

        return total_losses

    def semi_gradient_td0(self, env, policy, num_iterations=1000, gamma=0.99, alpha=0.1, summary_writer=None, summary_prefix=""):
        self.v = ApproximateValueFunction(env.observation_space.n, alpha)
        i = 0

        total_losses = []
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

                y = reward + gamma * self.v(new_state)
                self.v.append_x_y_pair(state, y)

                losses = self.v.approximate()
                if summary_writer is not None:
                    for l_idx, l in enumerate(losses):
                        summary_writer.add_scalar(f"{summary_prefix}loss", l, len(total_losses) + l_idx)

                total_losses.extend(losses)

                state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99, batch_size=32, alpha=0.01, summary_writer=None):
        if algorithm == OnPolicyAlgorithm.GRADIENT_MONTE_CARLO_PREDICTION:
            self.gradient_monte_carlo_prediction(env, policy, gamma=gamma, num_iterations=num_iterations, batch_size=batch_size, alpha=alpha, summary_writer=summary_writer)
        elif algorithm == OnPolicyAlgorithm.SEMI_GRADIENT_TD_0:
            self.semi_gradient_td0(env, policy, num_iterations=num_iterations, gamma=gamma, alpha=alpha, summary_writer=summary_writer)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def prediction():
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_name = "FrozenLake-v0"
    algorithm = OnPolicyAlgorithm.SEMI_GRADIENT_TD_0
    environment = gym.make(env_name)

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    k = 0
    gamma = 0.99
    agent = ApproximateAgent()
    num_iterations = 10 ** 3
    agent.predict(environment, policy, algorithm,gamma=gamma, num_iterations=num_iterations, summary_writer=writer)

    for i in range(environment.observation_space.n):
        print(f"V({i}) = {agent.v(str(i))}")

    print("")
    pass


if __name__ == "__main__":
    prediction()
