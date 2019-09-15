import collections
from enum import Enum

import torch
from torch import optim, nn
import gym
import gym.wrappers
import numpy as np
import re
from math import inf
import logging
from tensorboardX import SummaryWriter
from gym import envs

from core import TabularPolicy, EpisodeResult, EpsilonGreedyTabularPolicy, StateActionValueTable, CustomPolicy, \
    ApproximateValueFunction, ApproximateStateActionFunction, ApproximatePolicy

logger = logging.getLogger(__file__)


class PolicyGradientAlgorithm(Enum):
    REINFORCE = "REINFORCE"
    REINFORCE_WITH_BASELINE = "REINFORCE_WITH_BASELINE"


class PolicyGradientAgent(object):

    def __init__(self):
        self.v = None
        self.policy = None

    def play(self, env, num_episodes=100, gamma=0.99, render=False):
        i = 0
        best_return = float("-inf")
        best_result = None
        episode_returns = []
        while i < num_episodes:
            state = env.reset()
            done = False

            episode_result = EpisodeResult(env, state)
            while not done:
                if render:
                    env.render()

                action = self.policy(state)
                new_state, reward, done, info = env.step(action)

                episode_result.append(action, reward, new_state)

                state = new_state

            episode_return = episode_result.calculate_return(gamma)
            if best_return < episode_return:
                best_return = episode_return
                best_result = episode_result
                logger.info("New best return: {}".format(best_return))

            episode_returns.append(episode_return)
            i += 1

            logger.info(f"Episode Return: {episode_return}")
            logger.info(f"Episode Length: {len(episode_result.states)}")
        env.close()
        return episode_returns, best_result, best_return

    def reinforce(self, env, num_iterations=10000, batch_size=32, gamma=0.99, alpha=0.01,
                  summary_writer: SummaryWriter = None, summary_prefix=""):
        self.policy = ApproximatePolicy(env.observation_space, env.action_space.n, alpha)
        i = 0
        total_losses = []

        while i < num_iterations:
            state = env.reset()
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = self.policy(state)
                state, reward, done, _ = env.step(action)
                episode_result.append(action, reward, state)

            g = 0

            j = len(episode_result.states) - 2
            while j >= 0:
                g = gamma * g + episode_result.rewards[j]
                state = episode_result.states[j]
                action = episode_result.actions[j]

                discounted = gamma ** j * g
                self.policy.append(state, action, discounted)
                j -= 1

            if len(self.policy.state_batches) > batch_size:
                losses = self.policy.policy_gradient_approximation(batch_size)
                if summary_writer is not None:
                    for l_idx, l in enumerate(losses):
                        summary_writer.add_scalar(f"{summary_prefix}batch_loss", l, len(total_losses) + l_idx)
                total_losses.extend(losses)

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

        losses = self.policy.policy_gradient_approximation(batch_size)
        if summary_writer is not None:
            for l_idx, l in enumerate(losses):
                summary_writer.add_scalar(f"{summary_prefix}batch_loss", l, len(total_losses) + l_idx)
        total_losses.extend(losses)

        return total_losses

    # TODO test
    def reinforce_with_baseline(self, env, num_iterations=10000, batch_size=32, gamma=0.99, alpha=0.01,
                  summary_writer: SummaryWriter = None, summary_prefix=""):
        self.policy = ApproximatePolicy(env.observation_space, env.action_space.n, alpha)
        self.v = ApproximateValueFunction(env.observation_space, alpha)
        i = 0
        total_p_losses = []
        total_v_losses = []
        total_returns = []

        while i < num_iterations:
            state = env.reset()
            done = False
            episode_result = EpisodeResult(env, state)

            while not done:
                action = self.policy(state)
                state, reward, done, _ = env.step(action)
                episode_result.append(action, reward, state)

            g = 0

            # TODO test
            ep_return = episode_result.calculate_return(gamma)
            total_returns.append(ep_return)
            print(f"{i}: Length: {len(episode_result.states) - 1} \t Return: {ep_return:.2f} \t 100 Average: {np.array(total_returns[-100:]).mean():.2f}")

            j = len(episode_result.states) - 2
            while j >= 0:
                g = gamma * g + episode_result.rewards[j]
                v_state = self.v(state)
                delta = g - v_state
                state = episode_result.states[j]
                action = episode_result.actions[j]

                self.v.append_x_y_pair(state, g)
                discounted = gamma ** j * delta
                self.policy.append(state, action, discounted)
                # TODO test
                # self.policy.append(state, action, g - 5.0)
                # print(f"v: {v_state}")
                j -= 1

            if len(self.policy.state_batches) > batch_size:
                # TODO test
                p_losses = self.policy.policy_gradient_approximation(batch_size, mean_baseline=False)
                v_losses = self.v.approximate(batch_size)
                if summary_writer is not None:
                    for l_idx, l in enumerate(p_losses):
                        summary_writer.add_scalar(f"{summary_prefix}batch_policy_loss", l, len(total_p_losses) + l_idx)
                    for l_idx, l in enumerate(v_losses):
                        summary_writer.add_scalar(f"{summary_prefix}batch_value_loss", l, len(total_v_losses) + l_idx)
                total_p_losses.extend(p_losses)
                total_v_losses.extend(v_losses)

            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

        # TODO test
        p_losses = self.policy.policy_gradient_approximation(batch_size, mean_baseline=False)
        v_losses = self.v.approximate(batch_size)
        if summary_writer is not None:
            for l_idx, l in enumerate(p_losses):
                summary_writer.add_scalar(f"{summary_prefix}batch_policy_loss", l, len(total_p_losses) + l_idx)
            for l_idx, l in enumerate(v_losses):
                summary_writer.add_scalar(f"{summary_prefix}batch_value_loss", l, len(total_v_losses) + l_idx)
        total_p_losses.extend(p_losses)
        total_v_losses.extend(v_losses)

        return total_p_losses, total_v_losses

    def predict(self, env, policy, algorithm, num_iterations, n=2, gamma=0.99, batch_size=32, alpha=0.01, summary_writer=None):
        raise NotImplementedError()

    def learn(self, env, algorithm, num_iterations, gamma=0.99, batch_size=32, alpha=0.01, summary_writer=None):
        if algorithm == PolicyGradientAlgorithm.REINFORCE:
            self.reinforce(env, num_iterations=num_iterations, batch_size=batch_size, gamma=gamma, alpha=alpha,
                           summary_writer=summary_writer)
        elif algorithm == PolicyGradientAlgorithm.REINFORCE_WITH_BASELINE:
            self.reinforce_with_baseline(env, num_iterations=num_iterations, batch_size=batch_size, gamma=gamma,
                                         alpha=alpha,
                                         summary_writer=summary_writer)
        else:
            raise ValueError(f"Unknown algorithm {algorithm}")




def control():
    logging.basicConfig(level=logging.INFO)
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "CartPole-v0"
    # env_name = "FrozenLake-v0"
    algorithm = PolicyGradientAlgorithm.REINFORCE_WITH_BASELINE
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99
    learning_rate = 0.001

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 1000
    agent = PolicyGradientAgent()
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 10 ** 3
    num_test_episodes = 100
    batch_size = 16
    while True:
        agent.learn(environment, algorithm, num_iterations, gamma=gamma, batch_size=batch_size, alpha=learning_rate)
        round_test_returns, round_test_best_result, round_test_best_return = agent.play(test_env, render=True,
                                                                                        gamma=gamma,
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
    # x = gym.wrappers.Monitor()
    print("")
