import collections
from enum import Enum

import gym
import numpy as np
import re
from math import inf
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonSoftTabularPolicy, StateActionValueTable, CustomPolicy

logger = logging.getLogger(__file__)


class NStepAlgorithm(Enum):
    N_STEP_TD_PREDICTION = "N_STEP_TD_PREDICTION"


class NStepAgent(object):

    def __init__(self, n):
        if n <= 0:
            raise ValueError("N needs to be positive.")
        self.v_table = {}
        self.q_table = StateActionValueTable()
        self.n = n
        self.policy = None

    def n_step_td_prediction(self, env, policy, alpha=0.5, gamma=0.99, num_iterations=1000):
        self.v_table = collections.defaultdict(float)
        i = 0
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
                tau = j - self.n + 1
                if tau >= 0:
                    s_tau = episode_result.states[tau]
                    sum_up_to = min(tau + self.n, T)
                    k = tau + 1
                    g = 0.0
                    while k <= sum_up_to:
                        g += gamma ** (k - tau - 1) * episode_result.rewards[k - 1]  # TODO check indices
                        k += 1

                    if tau + self.n < T:
                        s_tau_n = episode_result.states[tau + self.n]
                        g += gamma ** self.n * self.v_table[s_tau_n]

                    self.v_table[s_tau] += alpha * (g - self.v_table[s_tau])

                j += 1
            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99):
        if algorithm == NStepAlgorithm.N_STEP_TD_PREDICTION:
            self.n_step_td_prediction(env, policy, gamma=gamma, num_iterations=num_iterations)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def prediction():
    policy = CustomPolicy.get_simple_blackjack_policy()
    env_name = "Blackjack-v0"
    algorithm = NStepAlgorithm.N_STEP_TD_PREDICTION
    environment = gym.make(env_name)

    k = 0
    gamma = 0.99
    n = 5
    agent = NStepAgent(2)
    num_iterations = 10000
    agent.predict(environment, policy, algorithm, gamma=gamma, num_iterations=num_iterations)

    print("")
    pass


if __name__ == "__main__":
    prediction()
