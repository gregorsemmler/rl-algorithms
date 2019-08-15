import collections
from enum import Enum

import gym
import numpy as np
import re
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import StateActionValueTable, EpisodeResult
from tabular.temporal_difference import TDAlgorithm

logger = logging.getLogger(__file__)


class ActionSelector(Enum):
    EPSILON_GREEDY = "EPSILON_GREEDY"


class TDAgent(object):
    discrete_action_space_pattern = re.compile(r"Discrete\(([0-9]+)\)")

    def __init__(self, default_q=0.0, possible_actions=()):
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.q_table = StateActionValueTable(default_value=default_q, possible_actions=possible_actions)
        self.q_table2 = None

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

    def _select_action(self, env, q_table, state, selector=ActionSelector.EPSILON_GREEDY, epsilon=0.5, q2=None):
        if selector != ActionSelector.EPSILON_GREEDY:
            raise RuntimeError("Method not implemented")

        if np.random.uniform() <= epsilon:
            return env.action_space.sample()
        else:
            action = q_table.get_best_action(state, q2)

            if action is None:
                action = env.action_space.sample()

        return action

    def _calculate_state_value(self, q_table, state, algorithm, action=None, epsilon=0.5, q_table2=None):
        if algorithm == TDAlgorithm.Q_LEARNING:
            # Get Qmax for Q Learning
            return q_table.get_q_max(state)
        if algorithm == TDAlgorithm.SARSA:
            return q_table[state, action]
        if algorithm == TDAlgorithm.EXPECTED_SARSA:
            q_values = q_table[state]
            q_values = sorted(q_values.items(), key=lambda entry: -entry[1])  # Entry with highest value is first
            q_max, q_max_val = q_values[0]
            other_q_values = q_values[1:]
            n = len(q_values)
            new_value = (1 - ((n - 1) * epsilon / n)) * q_max_val
            for other_q, other_q_val in other_q_values:
                new_value += epsilon * other_q_val / n
            return new_value
        if algorithm == TDAlgorithm.DOUBLE_Q_LEARNING:
            q1_best_action = q_table.get_best_action(state)
            return q_table2[state, q1_best_action]

        raise RuntimeError("Method not implemented: {}".format(algorithm))

    def learn(self, env, algorithm, num_episodes=100, epsilon=0.5, alpha=0.5, gamma=0.9):
        if algorithm == TDAlgorithm.DOUBLE_Q_LEARNING and self.q_table2 is None:
            self.q_table2 = StateActionValueTable()
        possible_actions = self._get_discrete_possible_actions(env)

        if possible_actions:
            self.q_table.possible_actions = possible_actions
            if self.q_table2 is not None:
                self.q_table2.possible_actions = possible_actions

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
                if algorithm == TDAlgorithm.Q_LEARNING or algorithm == TDAlgorithm.EXPECTED_SARSA:
                    action = self._select_action(env, self.q_table, state, selector=ActionSelector.EPSILON_GREEDY,
                                                 epsilon=epsilon)
                elif algorithm == TDAlgorithm.SARSA:
                    if new_action is None:
                        action = self._select_action(env, self.q_table, state, selector=ActionSelector.EPSILON_GREEDY,
                                                     epsilon=epsilon)
                    else:
                        action = new_action
                elif algorithm == TDAlgorithm.DOUBLE_Q_LEARNING:
                    action = self._select_action(env, self.q_table, state, selector=ActionSelector.EPSILON_GREEDY,
                                                 epsilon=epsilon, q2=self.q_table2)
                else:
                    raise RuntimeError("Method not implemented")

                new_state, reward, done, info = env.step(action)
                new_state = str(new_state)

                episode_result.append(action, reward, new_state)

                new_action = None
                if algorithm == TDAlgorithm.SARSA:
                    new_action = self._select_action(env, self.q_table, new_state,
                                                     selector=ActionSelector.EPSILON_GREEDY,
                                                     epsilon=epsilon)

                q1_prob = 1.0 if algorithm != TDAlgorithm.DOUBLE_Q_LEARNING else np.random.uniform()

                if q1_prob >= 0.5:
                    state_value = self._calculate_state_value(self.q_table, new_state, algorithm,
                                                              q_table2=self.q_table2, epsilon=epsilon,
                                                              action=new_action)
                    td_error = (reward + gamma * state_value - self.q_table[state, action])
                    new_q_val = self.q_table[state, action] + alpha * td_error
                    self.q_table[state, action] = new_q_val
                else:
                    state_value = self._calculate_state_value(self.q_table2, new_state, algorithm,
                                                              q_table2=self.q_table, epsilon=epsilon, action=new_action)
                    td_error = (reward + gamma * state_value - self.q_table2[state, action])
                    new_q_val = self.q_table2[state, action] + alpha * td_error
                    self.q_table2[state, action] = new_q_val

                state = new_state
            i += 1

            episode_return = episode_result.calculate_return(gamma)
            if best_return < episode_return:
                best_return = episode_return
                best_result = episode_result

            episode_returns.append(episode_return)

            if i % 100 == 0:
                logger.info("{} iterations done".format(i))

        return best_result, best_return

    def play(self, env, selector=ActionSelector.EPSILON_GREEDY, num_episodes=100, epsilon=0.5, gamma=0.9, render=False):
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

                action = self._select_action(env, self.q_table, state, selector=selector, epsilon=epsilon,
                                             q2=self.q_table2)

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
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "Taxi-v2"
    algorithm = TDAlgorithm.DOUBLE_Q_LEARNING
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)
    logger.info("test")
    k = 0
    epsilon = 1.0
    goal_returns = env_spec.reward_threshold
    gamma = 1.0

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 1000
    agent = TDAgent()
    best_result, best_return = None, float("-inf")
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_train_episodes = 100
    num_test_episodes = 10
    while True:
        round_best_result, round_best_return = agent.learn(environment, algorithm, epsilon=epsilon, gamma=gamma,
                                                           num_episodes=num_train_episodes)
        round_test_returns, round_test_best_result, round_test_best_return = agent.play(test_env, epsilon=0.0,
                                                                                        gamma=gamma,
                                                                                        num_episodes=num_test_episodes)
        for r_idx, r in enumerate(round_test_returns):
            writer.add_scalar("test_return", r, len(test_returns) + r_idx)

        test_returns.extend(round_test_returns)

        if best_return < round_best_return:
            best_return = round_best_return
            best_result = round_best_result

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
    main()
