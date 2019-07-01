# -*- coding: utf-8 -*-
import collections
from enum import Enum

import gym
import numpy as np
import re
import json
import heapq
import logging
# TODO queue has PriorityQueues
from tensorboardX import SummaryWriter

logger = logging.getLogger(__file__)


class ActionSelector(Enum):
    EPSILON_GREEDY = "EPSILON_GREEDY"


class TDAlgorithm(Enum):
    Q_LEARNING = "Q_LEARNING"
    EPSILON_GREEDY_SARSA = "EPSILON_GREEDY_SARSA"
    EPSILON_GREEDY_EXPECTED_SARSA = "EPSILON_GREEDY_EXPECTED_SARSA"


class EpisodeResult(object):

    def __init__(self, env, start_state):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []

    def append(self, action, reward, state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return


class StateActionValueTable(object):

    def __init__(self, default_value=0.0, possible_actions=()):
        self.q = {}
        self.default_value = default_value
        self.possible_actions = possible_actions

    def _init_state_if_not_set(self, state, action=None):
        if state not in self.q:
            self.q[state] = {a: self.default_value for a in self.possible_actions}
        if action not in self.q[state] and action is not None:
            self.q[state][action] = self.default_value

    def __setitem__(self, key, value):
        if type(key) is not tuple or len(key) != 2:
            raise RuntimeError("Expected state-action pair as key")
        state, action = key
        self._init_state_if_not_set(state)
        self.q[state][action] = value

    def __getitem__(self, item):
        if type(item) is not tuple:  # state was supplied
            self._init_state_if_not_set(item)
            return self.q[item]
        if type(item) is tuple and len(item) == 2:
            state, action = item
            self._init_state_if_not_set(state, action)
            return self.q[state][action]
        raise RuntimeError("Expected state or state-action pair as key")

    def get_q_max_pair(self, state):
        q_values = self.__getitem__(state)
        q_values = sorted(q_values.items(), key=lambda entry: -entry[1])  # Entry with highest value is first
        if len(q_values) == 0:
            return None
        return q_values[0]

    def get_q_max(self, state):
        return self.get_q_max_pair(state)[1]

    def get_best_action(self, state):
        return self.get_q_max_pair(state)[0]

    def to_json_file(self, filename):
        with open(filename, "w") as f:
            f.write(json.dumps(self.__dict__))

    def from_json_file(self, filename):
        with open(filename, "r") as f:
            content = json.load(f)
        self.q = content["q"]
        self.default_value = content["default_value"]
        self.possible_actions = content["possible_actions"]


class DiscreteAgent(object):
    discrete_action_space_pattern = re.compile(r"Discrete\(([0-9]+)\)")

    def __init__(self, default_q=0.0, possible_actions=()):
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.q_table = StateActionValueTable(default_value=default_q, possible_actions=possible_actions)
        self.q_table2 = None  # TODO for double q learning

    def _get_discrete_possible_actions(self, env):
        """
        Returns the possible actions of the supplied Environment if it has a discrete action space, else None
        :param env: the Environment for which the actions need to be considered
        :return: The possible actions as a list or None, if no actions could be detected
        """
        match = self.discrete_action_space_pattern.match(env.action_space.__repr__())
        if match:
            return list(range(int(match.group(1))))
        return None

    def _select_action(self, env, q_table, state, method=ActionSelector.EPSILON_GREEDY, epsilon=0.5):
        if method != ActionSelector.EPSILON_GREEDY:
            raise RuntimeError("Method not implemented")

        if np.random.uniform(1) <= epsilon:
            return env.action_space.sample()
        else:
            action = q_table.get_q_max_pair(state)[0]

            if action is None:
                action = env.action_space.sample()

        return action

    def _calculate_state_value(self, q_table, state, algorithm, action=None, epsilon=0.5):
        if algorithm == TDAlgorithm.Q_LEARNING:
            # Get Qmax for Q Learning
            return q_table.get_q_max(state)
        if algorithm == TDAlgorithm.EPSILON_GREEDY_SARSA:
            return q_table[state, action]
        if algorithm == TDAlgorithm.EPSILON_GREEDY_EXPECTED_SARSA:
            q_values = q_table[state]
            q_values = sorted(q_values.items(), key=lambda entry: -entry[1])  # Entry with highest value is first
            q_max, q_max_val = q_values[0]
            other_q_values = q_values[1:]
            n = len(q_values)
            new_value = (1 - ((n-1) * epsilon / n)) * q_max_val
            for other_q, other_q_val in other_q_values:
                new_value += epsilon * other_q_val / n
            return new_value

        raise RuntimeError("Method not implemented: {}".format(algorithm))

    def learn(self, env, algorithm, num_episodes=100, epsilon=0.5, alpha=0.5, gamma=0.9, *args, **kwargs):
        q = StateActionValueTable()
        possible_actions = self._get_discrete_possible_actions(env)

        if possible_actions:
            q.possible_actions = possible_actions

        episode_returns = []
        best_return = float("-inf")
        best_result = None

        i = 0
        while i < num_episodes:
            state = env.reset()
            state = str(state)
            done = False

            episode_result = EpisodeResult(env, state)
            new_action = None
            while not done:
                if algorithm == TDAlgorithm.Q_LEARNING or algorithm == TDAlgorithm.EPSILON_GREEDY_EXPECTED_SARSA:
                    action = self._select_action(env, q, state, method=ActionSelector.EPSILON_GREEDY, epsilon=epsilon)
                elif algorithm == TDAlgorithm.EPSILON_GREEDY_SARSA:
                    if new_action is None:
                        action = self._select_action(env, q, state, method=ActionSelector.EPSILON_GREEDY,
                                                     epsilon=epsilon)
                    else:
                        action = new_action
                else:
                    raise RuntimeError("Method not implemented")

                new_state, reward, done, info = env.step(action)
                new_state = str(new_state)

                episode_result.append(action, reward, new_state)

                new_action = None
                if algorithm == TDAlgorithm.EPSILON_GREEDY_SARSA:
                    new_action = self._select_action(env, q, new_state, method=ActionSelector.EPSILON_GREEDY,
                                                     epsilon=epsilon)

                state_value = self._calculate_state_value(q, new_state, algorithm, epsilon=epsilon, action=new_action)
                td_error = (reward + gamma * state_value - q[state, action])
                new_q_val = q[state, action] + alpha * td_error
                q[state, action] = new_q_val

                state = new_state
            i += 1

            episode_return = episode_result.calculate_return(gamma)
            if best_return < episode_return:
                best_return = episode_return
                best_result = episode_result

            episode_returns.append(episode_return)

            if i % 100 == 0:
                logger.info("{} iterations done".format(i))

        self.q_table = q
        return best_result, best_return

    def play(self, env, *args, **kwargs):
        pass

    # def select_best_action(self, state):
    #     best_action, best_value = None, None
    #     for action in range(self.env.action_space.n):
    #         action_value = self.q_values[(state, action)]
    #         if best_value is None or best_value < action_value:
    #             best_value = action_value
    #             best_action = action
    #     return best_action

    # def play_episode(self, env):
    #     total_reward = 0.0
    #     state = env.reset()
    #     while True:
    #         action = self.select_best_action(state)
    #         new_state, reward, is_done, _ = env.step(action)
    #         self.rewards[(state, action, new_state)] = reward
    #         self.transitions[(state, action)][new_state] += 1
    #         total_reward += reward
    #         if is_done:
    #             break
    #         state = new_state
    #     return total_reward
    #
    # def q_value_iteration(self, gamma):
    #     for state in range(self.env.observation_space.n):
    #         for action in range(self.env.action_space.n):
    #             action_value = 0.0
    #             target_counts = self.transitions[(state, action)]
    #             total = sum(target_counts.values())
    #             for tgt_state, count in target_counts.items():
    #                 reward = self.rewards[(state, action, tgt_state)]
    #                 best_action = self.select_best_action(tgt_state)
    #                 action_value += (count / total) * (reward + gamma * self.q_values[(tgt_state, best_action)])
    #             self.q_values[(state, action)] = action_value
    #
    # def play(self, gamma, num_random_steps=100, num_test_episodes=20):
    #     writer = SummaryWriter(comment="-q-iteration")
    #
    #     iter_no = 0
    #     best_reward = 0.0
    #     while True:
    #         iter_no += 1
    #         self.play_n_random_steps(num_random_steps)
    #         self.q_value_iteration(gamma)
    #
    #         reward = 0.0
    #         for _ in range(num_test_episodes):
    #             reward += self.play_episode(test_env)
    #         reward /= num_test_episodes
    #         writer.add_scalar("reward", reward, iter_no)
    #         if reward > best_reward:
    #             print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
    #             best_reward = reward
    #         if reward > 0.80:
    #             print("Solved in %d iterations!" % iter_no)
    #             break
    #     writer.close()


def simple_blackjack_policy(state):
    score, dealer_score, usable_ace = state
    if score <= 11:
        return 1
    return 0


def proportional_policy_from_q(q: StateActionValueTable, state):
    """Returns a policy which assigns a probability proportional to the state-action values for a given state"""
    pairs = sorted(q[state].items(), key=lambda e: e[1])
    if len(pairs) == 0:
        return None
    minimum_val = pairs[0][1]
    policy = {pair[0]: abs(minimum_val) + pair[1] for pair in pairs}  # shift so result is non negative
    total = sum(e[1] for e in policy.items())
    if total == 0.0:  # give equal probability to all actions
        return {pair[0]: 1 / len(pairs) for pair in pairs}
    policy = {pair[0]: pair[1] / total for pair in policy.items()}
    return policy


def sample_from_tabular_policy(policy):
    keys, values = list(zip(*sorted(policy.items(), key=lambda e: -e[1])))
    return keys[np.random.choice(len(keys), p=values)]


def epsilon_greedy_from_q(env: gym.Env, q: StateActionValueTable, state, epsilon):
    if np.random.uniform(1) <= epsilon:
        return env.action_space.sample()
    else:
        action = q.get_q_max_pair(state)[0]

        if action is None:
            action = env.action_space.sample()
    return action


# Estimates the value function of a given environment and policy
def tabular_td0(env, policy, alpha=0.01, gamma=0.99, num_iterations=10000):
    v = {}
    i = 0
    while i < num_iterations:
        state = env.reset()
        v[state] = 0.0
        done = False

        while not done:
            action = policy(state)
            new_state, reward, done, _ = env.step(action)

            if new_state not in v:
                v[new_state] = 0.0  # initialize

            v[state] = v[state] + alpha * (reward + gamma * v[new_state] - v[state])
        i += 1

        if i % 100 == 0:
            print("{} iterations done".format(i))
    return v


def epsilon_greedy_tabular_sarsa(env, epsilon=0.1, alpha=0.5, gamma=0.99, num_iterations=10 ** 5):
    q = StateActionValueTable()

    discrete_pattern = re.compile(r"Discrete\(([0-9]+)\)")
    match = discrete_pattern.match(env.action_space.__repr__())
    if match:
        q.possible_actions = list(range(int(match.group(1))))

    i = 0
    while i < num_iterations:
        state = env.reset()
        state = str(state)
        action = epsilon_greedy_from_q(env, q, state, epsilon)
        done = False

        while not done:
            new_state, reward, done, info = env.step(action)
            new_state = str(new_state)

            new_action = epsilon_greedy_from_q(env, q, new_state, epsilon)

            td_error = (reward + gamma * q[new_state, new_action] - q[state, action])
            new_q_val = q[state, action] + alpha * td_error
            q[state, action] = new_q_val

            state = new_state
            action = new_action
        i += 1

        if i % 100 == 0:
            print("{} iterations done".format(i))

    return q


def tabular_q_learning(env, epsilon=0.1, alpha=0.5, gamma=0.99, num_iterations=10 ** 3):
    q = StateActionValueTable()

    discrete_pattern = re.compile(r"Discrete\(([0-9]+)\)")
    match = discrete_pattern.match(env.action_space.__repr__())
    if match:
        q.possible_actions = list(range(int(match.group(1))))

    episode_returns = []
    best_return = float("-inf")
    best_result = None

    i = 0
    while i < num_iterations:
        state = env.reset()
        state = str(state)
        done = False

        episode_result = EpisodeResult(env, state)

        while not done:
            action = epsilon_greedy_from_q(env, q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            new_state = str(new_state)

            episode_result.append(action, reward, new_state)

            q_max = q.get_q_max_pair(new_state)[1]
            td_error = (reward + gamma * q_max - q[state, action])
            new_q_val = q[state, action] + alpha * td_error
            q[state, action] = new_q_val

            state = new_state
        i += 1

        episode_return = episode_result.calculate_return(gamma)
        if best_return < episode_return:
            best_return = episode_return
            best_result = episode_result
            logger.info("New best return: {}".format(best_return))

        episode_returns.append(episode_return)

        if i % 100 == 0:
            logger.info("{} iterations done".format(i))

    return q, episode_returns, best_result


def tabular_expected_sarsa(env, alpha=0.5, gamma=0.99, num_iterations=10 ** 5):
    """Tabular Expected Sarsa as described in
    'A Theoretical and Empirical Analysis of Expected Sarsa' by van Seijen et al. (2009)"""
    q = StateActionValueTable()

    discrete_pattern = re.compile(r"Discrete\(([0-9]+)\)")
    match = discrete_pattern.match(env.action_space.__repr__())
    if match:
        q.possible_actions = list(range(int(match.group(1))))

    i = 0
    while i < num_iterations:
        state = env.reset()
        state = str(state)
        done = False

        while not done:
            policy = proportional_policy_from_q(q, state)
            action = sample_from_tabular_policy(policy)

            new_state, reward, done, info = env.step(action)
            new_state = str(new_state)

            value_new_state = 0.0
            policy_new_state = proportional_policy_from_q(q, new_state)
            for new_action, probability in policy_new_state.items():
                value_new_state += q[new_state, new_action] * policy_new_state[new_action]

            td_error = (reward + gamma * value_new_state - q[state, action])
            new_q_val = q[state, action] + alpha * td_error
            q[state, action] = new_q_val

            state = new_state
        i += 1

        if i % 100 == 0:
            print("{} iterations done".format(i))

    return q


def test_tabular_q_policy(env, q: StateActionValueTable, gamma=0.99, num_iterations=100, render=False, greedy=True):
    i = 0
    best_return = float("-inf")
    best_result = None
    episode_returns = []
    while i < num_iterations:
        state = env.reset()
        state = str(state)
        done = False

        episode_result = EpisodeResult(env, state)
        while not done:
            if render:
                env.render()

            if greedy:
                action = q.get_q_max_pair(state)[0]
            else:
                policy = proportional_policy_from_q(q, state)
                action = sample_from_tabular_policy(policy)

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


def main():
    env_name = "FrozenLake-v0"
    environment = gym.make(env_name)
    test_env = gym.make(env_name)
    logger.info("test")

    # if type(environment) == gym.wrappers.time_limit.TimeLimit:
    #     environment = environment.env

    # q = tabular_expected_sarsa(environment)
    # q = epsilon_greedy_tabular_sarsa(environment)
    epsilon = 1.0

    k = 0
    epsilon = 1.0
    goal_returns = 0.8

    agent = DiscreteAgent()
    while True:
        # q, returns, best_result = tabular_q_learning(environment, epsilon=epsilon)
        best_result, best_return = agent.learn(environment, TDAlgorithm.EPSILON_GREEDY_EXPECTED_SARSA, epsilon=1.0,
                                               num_episodes=10 ** 3)
        q = agent.q_table
        test_returns, test_best_result, test_best_return = test_tabular_q_policy(test_env, q, greedy=True,
                                                                                 num_iterations=20)

        average_test_returns = 0.0 if len(test_returns) == 0 else sum(test_returns) / len(test_returns)
        logger.warning("Average returns: {}".format(average_test_returns))

        if average_test_returns >= goal_returns:
            logger.warning("Done in {} rounds!".format(k))
            break
        k += 1

    print("")
    pass


if __name__ == "__main__":
    main()
