import collections
from enum import Enum

import torch
from torch import optim, nn
import gym
import numpy as np
import re
from math import inf
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonGreedyTabularPolicy, StateActionValueTable, CustomPolicy, \
    ApproximateValueFunction

logger = logging.getLogger(__file__)


class OnPolicyAlgorithm(Enum):
    GRADIENT_MONTE_CARLO_PREDICTION = "GRADIENT_MONTE_CARLO_PREDICTION"
    SEMI_GRADIENT_TD_0 = "SEMI_GRADIENT_TD_0"
    N_STEP_SEMI_GRADIENT_TD = "N_STEP_SEMI_GRADIENT_TD"


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

            if len(self.v.x_batches) > batch_size:
                losses = self.v.approximate(batch_size)
                if summary_writer is not None:
                    for l_idx, l in enumerate(losses):
                        summary_writer.add_scalar(f"{summary_prefix}batch_loss", l, len(total_losses) + l_idx)
                total_losses.extend(losses)

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

        losses = self.v.approximate(batch_size)
        if summary_writer is not None:
            for l_idx, l in enumerate(losses):
                summary_writer.add_scalar(f"{summary_prefix}batch_loss", l, len(total_losses) + l_idx)
        total_losses.extend(losses)

        return total_losses

    def semi_gradient_td0(self, env, policy, num_iterations=1000, gamma=0.99, alpha=0.1, batch_size=32, summary_writer=None, summary_prefix=""):
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

                if len(self.v.x_batches) > batch_size:
                    losses = self.v.approximate(batch_size)
                    if summary_writer is not None:
                        for l_idx, l in enumerate(losses):
                            summary_writer.add_scalar(f"{summary_prefix}loss", l, len(total_losses) + l_idx)
                        logger.info(f"{sum(losses)/max(1, len(losses))}")
                    total_losses.extend(losses)

                state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

        losses = self.v.approximate(batch_size)
        if summary_writer is not None:
            for l_idx, l in enumerate(losses):
                summary_writer.add_scalar(f"{summary_prefix}batch_loss", l, len(total_losses) + l_idx)
        total_losses.extend(losses)


    def n_step_semi_gradient_td(self, env, n, policy, alpha=0.01, gamma=0.99, num_iterations=1000, batch_size=32, summary_writer=None, summary_prefix=""):
        self.v = ApproximateValueFunction(env.observation_space.n, alpha)
        i = 0

        total_losses = []
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            episode_result = EpisodeResult(env, state)
            T = inf
            tau = -inf

            j = 0
            while tau != T - 1:
                if j < T:
                    action = policy(state)
                    state, reward, done, _ = env.step(action)
                    state = str(state)
                    episode_result.append(action, reward, state)

                    if done:
                        T = j + 1
                tau = j - n + 1
                if tau >= 0:
                    s_tau = episode_result.states[tau]
                    sum_up_to = min(tau + n, T)
                    k = tau + 1
                    g = 0.0
                    while k <= sum_up_to:
                        g += gamma ** (k - tau - 1) * episode_result.rewards[k - 1]
                        k += 1

                    if tau + n < T:
                        s_tau_n = episode_result.states[tau + n]
                        g += gamma ** n * self.v(s_tau_n)

                    self.v.append_x_y_pair(s_tau, g)
                    if len(self.v.x_batches) > batch_size:
                        losses = self.v.approximate(batch_size)
                        if summary_writer is not None:
                            for l_idx, l in enumerate(losses):
                                summary_writer.add_scalar(f"{summary_prefix}loss", l, len(total_losses) + l_idx)
                            logger.info(f"{sum(losses)/max(1, len(losses))}")
                        total_losses.extend(losses)

                j += 1
            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        losses = self.v.approximate(batch_size)
        if summary_writer is not None:
            for l_idx, l in enumerate(losses):
                summary_writer.add_scalar(f"{summary_prefix}loss", l, len(total_losses) + l_idx)
            logger.info(f"{sum(losses) / max(1, len(losses))}")
        total_losses.extend(losses)
        return total_losses

    def predict(self, env, policy, algorithm, num_iterations, n=2, gamma=0.99, batch_size=32, alpha=0.01, summary_writer=None):
        if algorithm == OnPolicyAlgorithm.GRADIENT_MONTE_CARLO_PREDICTION:
            self.gradient_monte_carlo_prediction(env, policy, gamma=gamma, num_iterations=num_iterations, batch_size=batch_size, alpha=alpha, summary_writer=summary_writer)
        elif algorithm == OnPolicyAlgorithm.SEMI_GRADIENT_TD_0:
            self.semi_gradient_td0(env, policy, num_iterations=num_iterations, gamma=gamma, alpha=alpha,
                                   batch_size=batch_size, summary_writer=summary_writer)
        elif algorithm == OnPolicyAlgorithm.N_STEP_SEMI_GRADIENT_TD:
            self.n_step_semi_gradient_td(env, n, policy, alpha=alpha, gamma=gamma, num_iterations=num_iterations, batch_size=batch_size, summary_writer=summary_writer)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def prediction():
    # logging.basicConfig(level=logging.DEBUG)
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_name = "FrozenLake-v0"
    algorithm = OnPolicyAlgorithm.N_STEP_SEMI_GRADIENT_TD
    environment = gym.make(env_name)

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    gamma = 0.99
    agent = ApproximateAgent()
    num_iterations = 10 ** 4
    batch_size = 16
    n = 3
    alpha = 0.01
    agent.predict(environment, policy, algorithm, n=n, alpha=alpha, gamma=gamma, num_iterations=num_iterations, batch_size=batch_size,
                  summary_writer=writer)

    for i in range(environment.observation_space.n):
        print(f"V({i}) = {agent.v(str(i))}")

    print("")
    pass


if __name__ == "__main__":
    prediction()
