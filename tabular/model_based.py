import collections
from enum import Enum

import gym
import numpy as np
import re
from math import inf
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import SampleEnvironmentModel, TabularPolicy, StateActionValueTable, EpsilonGreedyTabularPolicy, EpisodeResult

logger = logging.getLogger(__file__)


class ModelAlgorithm(Enum):
    RANDOM_ONE_STEP_Q_PLANNING = "RANDOM_ONE_STEP_Q_PLANNING"


class TabularModelAgent(object):

    def __init__(self, model: SampleEnvironmentModel):
        self.model = model
        self.q_table = None
        self.policy = None

    def one_step_q_planning(self, env, num_iterations, num_exploration_steps=1000, gamma=0.99, alpha=0.5, epsilon=0.1,
                            b=None):
        self.q_table = StateActionValueTable(possible_actions=range(env.action_space.n))
        self.policy = EpsilonGreedyTabularPolicy(env.action_space.n, epsilon=epsilon)

        self.model.estimate(env, b=b, num_iterations=num_exploration_steps)

        i = 0
        while i < num_iterations:
            sampled = None

            while sampled is None:
                state, action = self.model.random_state_and_action()
                sampled = self.model.sample(state, action)

            new_state, reward = sampled

            update = reward + gamma * self.q_table.get_q_max(state) - self.q_table[state, action]
            update *= alpha
            self.q_table[state, action] += update
            self.policy[state] = self.q_table.get_best_action(state)
            state = new_state

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))
        pass

    def learn(self, env, algorithm, num_iterations, num_exploration_steps=1000, gamma=0.99, alpha=0.5, epsilon=0.1,
              b=None):
        if algorithm == ModelAlgorithm.RANDOM_ONE_STEP_Q_PLANNING:
            self.one_step_q_planning(env, num_iterations, num_exploration_steps=num_exploration_steps, gamma=gamma,
                                     alpha=alpha, epsilon=epsilon, b=b)
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
    env_name = "FrozenLake-v0"
    algorithm = ModelAlgorithm.RANDOM_ONE_STEP_Q_PLANNING
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99
    alpha = 0.5
    epsilon = 0.5

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 1000

    model = SampleEnvironmentModel()
    agent = TabularModelAgent(model)

    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 1 * 10 ** 3
    num_exploration_steps = 2000
    num_test_episodes = 100
    while True:
        agent.learn(environment, algorithm, num_iterations, num_exploration_steps=num_exploration_steps, gamma=gamma,
                    alpha=alpha, epsilon=epsilon)
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
