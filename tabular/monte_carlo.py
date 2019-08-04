import collections
from enum import Enum

import gym
import numpy as np
import re
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonSoftTabularPolicy, StateActionValueTable, CustomPolicy

logger = logging.getLogger(__file__)


class MCAlgorithm(Enum):
    MC_FIRST_VISIT_PREDICTION = "MONTE_CARLO_FIRST_VISIT_PREDICTION"
    MC_EVERY_VISIT_PREDICTION = "MONTE_CARLO_EVERY_VISIT_PREDICTION"
    OFF_POLICY_MC_PREDICTION = "OFF_POLICY_MONTE_CARLO_PREDICTION"
    ON_POLICY_FIRST_VISIT_MC_CONTROL = "ON_POLICY_FIRST_VISIT_MC_CONTROL"
    OFF_POLICY_MC_CONTROL = "OFF_POLICY_MONTE_CARLO_CONTROL"


class MonteCarloAgent(object):

    def __init__(self):
        self.v_table = {}
        self.q_table = StateActionValueTable()
        self.policy = None

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

    def off_policy_mc_prediction(self, env, policy, gamma=0.99, num_iterations=1000, b=None):
        if b is None:
            behavior_policy = EpsilonSoftTabularPolicy(range(env.action_space.n), 1.0)  # Use random policy by default
        else:
            behavior_policy = b

        self.q_table = StateActionValueTable()
        c = collections.defaultdict(float)

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = behavior_policy(state)
                state, reward, done, _ = env.step(action)
                state = str(state)
                episode_result.append(action, reward, state)

            g = 0
            w = 1.0

            j = len(episode_result.states) - 2
            while j >= 0 and w != 0:
                g = gamma * g + episode_result.rewards[j]
                state = episode_result.states[j]
                action = episode_result.actions[j]
                c[(state, action)] = c[(state, action)] + w
                addition = w / c[(state, action)] * (g - self.q_table[state, action])
                self.q_table[state, action] = self.q_table[state, action] + addition
                policy_prob = policy.get_probability(action, state)
                behavior_prob = behavior_policy.get_probability(action, state)
                w = w * (policy_prob / behavior_prob)
                j -= 1

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99, b=None):
        if algorithm == MCAlgorithm.MC_FIRST_VISIT_PREDICTION:
            self.monte_carlo_prediction(env, policy, first_visit=True, gamma=gamma, num_iterations=num_iterations)
        elif algorithm == MCAlgorithm.MC_EVERY_VISIT_PREDICTION:
            self.monte_carlo_prediction(env, policy, first_visit=False, gamma=gamma, num_iterations=num_iterations)
        elif algorithm == MCAlgorithm.OFF_POLICY_MC_PREDICTION:
            self.off_policy_mc_prediction(env, policy, b=b, gamma=gamma, num_iterations=num_iterations)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))

    def on_policy_first_visit_mc_control(self, env, epsilon, gamma=0.99, num_iterations=1000):
        self.policy = EpsilonSoftTabularPolicy(range(env.action_space.n), epsilon)
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        returns = collections.defaultdict(list)

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)
            first_visited = {state: 0}

            while not done:
                action = self.policy(state)
                state, reward, done, _ = env.step(action)
                state = str(state)
                episode_result.append(action, reward, state)
                if state not in first_visited:
                    first_visited[state] = len(episode_result.rewards)

            g = 0
            j = len(episode_result.states) - 2
            while j >= 0:
                g = gamma * g + episode_result.rewards[j]
                state = episode_result.states[j]
                action = episode_result.actions[j]
                if state in first_visited and first_visited[state] == j:
                    returns[(state, action)].append(g)
                    self.q_table[state, action] = sum(returns[(state, action)]) / len(returns[(state, action)])
                    self.policy[state] = self.q_table.get_best_action(state)
                j -= 1

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def off_policy_mc_control(self, env, gamma=0.99, num_iterations=1000, b=None, q=None):
        if b is None:
            behavior_policy = EpsilonSoftTabularPolicy(range(env.action_space.n), 1.0)  # Use random policy by default
        else:
            behavior_policy = b

        if q is None:
            self.q_table = StateActionValueTable()
        else:
            self.q_table = q

        c = collections.defaultdict(float)
        self.policy = TabularPolicy.greedy_from_q_table(self.q_table)

        i = 0
        while i < num_iterations:
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = behavior_policy(state)
                state, reward, done, _ = env.step(action)
                state = str(state)
                episode_result.append(action, reward, state)

            g = 0
            w = 1.0

            j = len(episode_result.states) - 2
            while j >= 0:
                g = gamma * g + episode_result.rewards[j]
                state = episode_result.states[j]
                action = episode_result.actions[j]
                c[(state, action)] = c[(state, action)] + w
                addition = w / c[(state, action)] * (g - self.q_table[state, action])
                self.q_table[state, action] = self.q_table[state, action] + addition
                self.policy[state] = self.q_table.get_best_action(state)
                if action != self.policy[state]:
                    break
                w /= behavior_policy.get_probability(action, state)
                j -= 1

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def learn(self, env, algorithm, epsilon, gamma=0.99, num_iterations=1000, b=None, q=None):
        if algorithm == MCAlgorithm.ON_POLICY_FIRST_VISIT_MC_CONTROL:
            self.on_policy_first_visit_mc_control(env, epsilon, gamma=gamma, num_iterations=num_iterations)
        elif algorithm == MCAlgorithm.OFF_POLICY_MC_CONTROL:
            self.off_policy_mc_control(env, gamma=gamma, num_iterations=num_iterations, b=b, q=q)
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
    algorithm = MCAlgorithm.OFF_POLICY_MC_PREDICTION
    environment = gym.make(env_name)

    k = 0
    gamma = 0.99
    agent = MonteCarloAgent()
    num_iterations = 10000
    agent.predict(environment, policy, algorithm,gamma=gamma, num_iterations=num_iterations)

    print("")
    pass


def control():
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "FrozenLake-v0"
    algorithm = MCAlgorithm.OFF_POLICY_MC_CONTROL
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
    num_iterations = 1000
    num_test_episodes = 100
    epsilon = 0.6
    while True:
        agent.learn(environment, algorithm, epsilon, gamma=gamma, num_iterations=num_iterations)
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
