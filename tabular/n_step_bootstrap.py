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


class NStepAlgorithm(Enum):
    N_STEP_TD_PREDICTION = "N_STEP_TD_PREDICTION"
    N_STEP_SARSA = "N_STEP_SARSA"


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
                        g += gamma ** (k - tau - 1) * episode_result.rewards[k - 1]
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

    def n_step_td_sarsa(self, env, epsilon, alpha=0.5, gamma=0.99, num_iterations=1000, q=None, policy=None):
        if q is None:
            self.q_table = StateActionValueTable()
        else:
            self.q_table = q

        if policy is None:
            self.policy = EpsilonGreedyTabularPolicy.from_q(env.action_space.n, epsilon, self.q_table)
        else:
            self.policy = policy

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            episode_result = EpisodeResult(env, state)
            T = inf
            tau = -inf

            # Choose first action without taking it
            action = self.policy(state)
            episode_result.actions.append(action)

            j = 0
            while tau != T - 1:
                if j < T:
                    state, reward, done, _ = env.step(action)
                    state = str(state)
                    episode_result.states.append(state)
                    episode_result.rewards.append(reward)

                    action = self.policy(state)
                    episode_result.actions.append(action)

                    if done:
                        T = j + 1
                else:
                    action = self.policy(state)
                    episode_result.actions.append(action)

                tau = j - self.n + 1
                if tau >= 0:
                    s_tau = episode_result.states[tau]
                    a_tau = episode_result.actions[tau]
                    sum_up_to = min(tau + self.n, T)
                    k = tau + 1
                    g = 0.0
                    while k <= sum_up_to:
                        g += gamma ** (k - tau - 1) * episode_result.rewards[k - 1]
                        k += 1

                    if tau + self.n < T:
                        s_tau_n = episode_result.states[tau + self.n]
                        a_tau_n = episode_result.actions[tau + self.n]
                        g += gamma ** self.n * self.q_table[s_tau_n, a_tau_n]

                    self.q_table[s_tau, a_tau] += alpha * (g - self.q_table[s_tau, a_tau])
                    self.policy[s_tau] = self.q_table.get_best_action(s_tau)

                j += 1
            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99):
        if algorithm == NStepAlgorithm.N_STEP_TD_PREDICTION:
            self.n_step_td_prediction(env, policy, gamma=gamma, num_iterations=num_iterations)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))

    def learn(self, env, algorithm, epsilon, alpha=0.5, gamma=0.99, num_iterations=1000, q=None, policy=None):
        if algorithm == NStepAlgorithm.N_STEP_SARSA:
            self.n_step_td_sarsa(env, epsilon, alpha=alpha, policy=policy, q=q, gamma=gamma, num_iterations=num_iterations)
        else:
            raise ValueError("Unknown Algorithm {}".format(algorithm))

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


def control():
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "Taxi-v2"
    algorithm = NStepAlgorithm.N_STEP_SARSA
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    n = 8
    max_rounds = 10000
    agent = NStepAgent(n)
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 1000
    num_test_episodes = 100
    epsilon = 0.1
    alpha = 0.2
    while True:
        agent.learn(environment, algorithm, epsilon, alpha=alpha, gamma=gamma, num_iterations=num_iterations)
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

    pass


if __name__ == "__main__":
    control()
