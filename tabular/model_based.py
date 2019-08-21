import collections
from enum import Enum

import gym
import numpy as np
import re
from math import inf, sqrt
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import SampleEnvironmentModel, TabularPolicy, StateActionValueTable, EpsilonGreedyTabularPolicy, EpisodeResult

logger = logging.getLogger(__file__)


class TabularModelAlgorithm(Enum):
    RANDOM_ONE_STEP_Q_PLANNING = "RANDOM_ONE_STEP_Q_PLANNING"
    TABULAR_DYNA_Q = "TABULAR_DYNA_Q"
    TABULAR_DYNA_Q_PLUS = "TABULAR_DYNA_Q_PLUS"


class TabularModelAgent(object):

    def __init__(self, model: SampleEnvironmentModel):
        self.model = model
        self.q_table = None
        self.policy = None

    def one_step_q_planning(self, env, num_iterations, num_exploration_steps=1000, gamma=0.99, alpha=0.5, epsilon=0.1,
                            exp_policy=None):
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon=epsilon)

        self.model.estimate(env, exp_policy=exp_policy, num_iterations=num_exploration_steps)

        i = 0
        while i < num_iterations:
            sampled = None

            while sampled is None:
                state, action = self.model.random_state_and_action()
                sampled = self.model.sample(state, action)

            new_state, reward = sampled

            update = reward + gamma * self.q_table.get_q_max(new_state) - self.q_table[state, action]
            update *= alpha
            self.q_table[state, action] += update
            self.policy[state] = self.q_table.get_best_action(state)
            state = new_state

            i += 1
        pass

    def tabular_dyna_q(self, env, num_iterations, num_exploration_steps=1000, gamma=0.99, alpha=0.5, epsilon=0.1,
                       exp_policy=None, samples_per_step=3, b=None, dyna_q_plus=False, kappa=0.5):
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon=epsilon)

        self.model.estimate(env, exp_policy=exp_policy, num_iterations=num_exploration_steps)

        if b is None:
            behavior_policy = EpsilonGreedyTabularPolicy.random_policy(env.action_space.n)
        else:
            behavior_policy = b

        i = 0
        while i < num_iterations:

            last_sampled = {}
            state = env.reset()
            state = str(state)
            done = False
            episode_result = EpisodeResult(env, state)

            episode_t = 0
            while not done:
                action = behavior_policy(state)
                new_state, reward, done, _ = env.step(action)
                new_state = str(new_state)
                episode_result.append(action, reward, new_state)
                self.model.append(state, action, new_state, reward, done)

                update = reward + gamma * self.q_table.get_q_max(new_state) - self.q_table[state, action]
                update *= alpha
                self.q_table[state, action] += update
                self.policy[state] = self.q_table.get_best_action(state)
                state = new_state

                j = 0

                while j < samples_per_step:
                    sampled = None

                    while sampled is None:
                        s_state, s_action = self.model.random_state_and_action()
                        sampled = self.model.sample(s_state, s_action)

                    s_new_state, s_reward = sampled
                    if sampled not in last_sampled:
                        last_sampled[sampled] = 0

                    if dyna_q_plus:
                        reward_bonus = kappa * sqrt(max(0, episode_t - last_sampled[sampled]))
                    else:
                        reward_bonus = 0.0
                    update = s_reward + reward_bonus + gamma * self.q_table.get_q_max(s_new_state) - self.q_table[s_state, s_action]
                    update *= alpha
                    self.q_table[s_state, s_action] += update
                    self.policy[s_state] = self.q_table.get_best_action(s_state)
                    s_state = s_new_state

                    last_sampled[sampled] = episode_t

                    j += 1
                episode_t += 1
            i += 1

    def learn(self, env, algorithm, num_iterations, num_exploration_steps=1000, gamma=0.99, alpha=0.5, epsilon=0.1,
              exp_policy=None, samples_per_step=3, b=None, kappa=0.5):
        if algorithm == TabularModelAlgorithm.RANDOM_ONE_STEP_Q_PLANNING:
            self.one_step_q_planning(env, num_iterations, num_exploration_steps=num_exploration_steps, gamma=gamma,
                                     alpha=alpha, epsilon=epsilon, exp_policy=exp_policy)
        elif algorithm == TabularModelAlgorithm.TABULAR_DYNA_Q:
            self.tabular_dyna_q(env, num_iterations, num_exploration_steps=num_exploration_steps, gamma=gamma,
                                alpha=alpha, epsilon=epsilon, exp_policy=exp_policy, samples_per_step=samples_per_step,
                                b=b)
        elif algorithm == TabularModelAlgorithm.TABULAR_DYNA_Q_PLUS:
            self.tabular_dyna_q(env, num_iterations, num_exploration_steps=num_exploration_steps, gamma=gamma,
                                alpha=alpha, epsilon=epsilon, exp_policy=exp_policy, samples_per_step=samples_per_step,
                                b=b, dyna_q_plus=True, kappa=kappa)
        else:
            raise ValueError("Unknown Learn Algorithm: {}".format(algorithm))

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


def control():
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "Taxi-v2"
    algorithm = TabularModelAlgorithm.TABULAR_DYNA_Q_PLUS
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

    model = SampleEnvironmentModel()
    agent = TabularModelAgent(model)

    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 100
    num_exploration_steps = 2000
    num_test_episodes = 100
    samples_per_step = 10
    kappa = 0.001
    while True:
        agent.learn(environment, algorithm, num_iterations, num_exploration_steps=num_exploration_steps, gamma=gamma,
                    alpha=alpha, epsilon=epsilon, samples_per_step=samples_per_step, kappa=kappa)
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
