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
    SARSA = "SARSA"
    EXPECTED_SARSA = "EXPECTED_SARSA"
    DOUBLE_Q_LEARNING = "DOUBLE_Q_LEARNING"


class TDAgent(object):

    def __init__(self):
        self.v_table = None
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

    def tabular_sarsa(self, env, policy=None, num_iterations=1000, gamma=0.99, alpha=0.5, epsilon=0.1):
        self.q_table = StateActionValueTable()

        if policy is not None:
            self.policy = policy
        else:
            self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon)

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            # choose first action without taking it
            action = self.policy(state)

            while not done:
                new_state, reward, done, _ = env.step(action)
                new_state = str(new_state)
                episode_result.append(action, reward, new_state)

                new_action = self.policy(new_state)

                update = reward + gamma * self.q_table[new_state, new_action] - self.q_table[state, action]
                update *= alpha
                self.q_table[state, action] += update
                self.policy[state] = self.q_table.get_best_action(state)
                state = new_state
                action = new_action

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def q_learning(self, env, num_iterations=1000, gamma=0.99, alpha=0.5, epsilon=0.1, b=None):
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon=epsilon)

        if b is not None:
            behavior_policy = b
        else:
            behavior_policy = EpsilonGreedyTabularPolicy.random_policy(env.action_space.n)

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = behavior_policy(state)
                new_state, reward, done, _ = env.step(action)
                new_state = str(new_state)
                episode_result.append(action, reward, new_state)

                update = reward + gamma * self.q_table.get_q_max(new_state) - self.q_table[state, action]
                update *= alpha
                self.q_table[state, action] += update
                self.policy[state] = self.q_table.get_best_action(state)
                state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def expected_sarsa(self, env, num_iterations=1000, gamma=0.99, alpha=0.5, epsilon=0.1, b=None):
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon=epsilon)

        if b is not None:
            behavior_policy = b
        else:
            behavior_policy = self.policy

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = behavior_policy(state)
                new_state, reward, done, _ = env.step(action)
                new_state = str(new_state)
                episode_result.append(action, reward, new_state)

                add = 0.0
                for a in range(env.action_space.n):
                    add += self.policy.get_probability(a, new_state) * self.q_table[new_state, a]

                update = reward + gamma * add - self.q_table[state, action]
                update *= alpha
                self.q_table[state, action] += update
                self.policy[state] = self.q_table.get_best_action(state)
                state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def double_q_learning(self, env, num_iterations=1000, gamma=0.99, alpha=0.5, epsilon=0.1, b=None):
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.q_table2 = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon)

        if b is not None:
            behavior_policy = b
        else:
            behavior_policy = self.policy

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = behavior_policy(state)
                new_state, reward, done, _ = env.step(action)
                new_state = str(new_state)
                episode_result.append(action, reward, new_state)

                if np.random.rand() <= 0.5:
                    a = self.q_table.get_best_action(new_state)
                    update = reward + gamma * self.q_table2[new_state, a] - self.q_table[state, action]
                    update *= alpha
                    self.q_table[state, action] += update
                else:
                    a = self.q_table2.get_best_action(new_state)
                    update = reward + gamma * self.q_table[new_state, a] - self.q_table2[state, action]
                    update *= alpha
                    self.q_table2[state, action] += update

                self.policy[state] = self.q_table.get_best_action(state, q2=self.q_table2)
                state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def learn(self, env, algorithm, num_iterations, policy=None, gamma=0.99, alpha=0.5, epsilon=0.1, b=None):
        if algorithm == TDAlgorithm.SARSA:
            self.tabular_sarsa(env, policy=policy, num_iterations=num_iterations, gamma=gamma, alpha=alpha,
                               epsilon=epsilon)
        elif algorithm == TDAlgorithm.Q_LEARNING:
            self.q_learning(env, num_iterations=num_iterations, gamma=gamma, alpha=alpha, epsilon=epsilon, b=b)
        elif algorithm == TDAlgorithm.EXPECTED_SARSA:
            self.expected_sarsa(env, num_iterations=num_iterations, gamma=gamma, alpha=alpha, epsilon=epsilon, b=b)
        elif algorithm == TDAlgorithm.DOUBLE_Q_LEARNING:
            self.expected_sarsa(env, num_iterations=num_iterations, gamma=gamma, alpha=alpha, epsilon=epsilon, b=b)
        else:
            raise ValueError("Unknown Learn Algorithm: {}".format(algorithm))


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
    env_name = "Taxi-v2"
    algorithm = TDAlgorithm.DOUBLE_Q_LEARNING
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99
    alpha = 0.5
    epsilon = 0.01

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 1000
    agent = TDAgent()
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 5 * 10**3
    num_test_episodes = 100
    while True:
        agent.learn(environment, algorithm, num_iterations, gamma=gamma, alpha=alpha, epsilon=epsilon)
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
    control()
