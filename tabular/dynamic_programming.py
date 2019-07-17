# -*- coding: utf-8 -*-
import collections
import logging
from enum import Enum

from tensorboardX import SummaryWriter

import gym
from gym import envs

from core import EpisodeResult

logger = logging.getLogger(__file__)


class DPAlgorithm(Enum):
    POLICY_ITERATION = "POLICY_ITERATION"
    VALUE_ITERATION = "VALUE_ITERATION"


class DPAgent(object):

    def __init__(self):
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.probabilities = collections.defaultdict(float)
        self.policy = collections.defaultdict(int)
        self.value_table = collections.defaultdict(float)

    def _select_action(self, env, state):
        return self.policy[state]

    def estimate_transition_probabilities(self, env, num_iterations=100):
        """
        Performs a number of random actions and estimates the state-action-state probabilities depending on how often
        they occurred during this time.
        :param env: The environment to be evaluated
        :param num_iterations: how many steps to perform
        :return:
        """
        state = env.reset()
        for _ in range(num_iterations):
            action = env.action_space.sample()
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(str(state), action, str(new_state))] = reward
            self.transitions[(str(state), action)][str(new_state)] += 1
            state = env.reset() if is_done else new_state

        for s, a in self.transitions.keys():
            num_transitions = sum(self.transitions[(s, a)].values())

            for s2 in self.transitions[(s, a)].keys():
                if num_transitions == 0:
                    self.probabilities[(s, a, s2)] = 0.0
                else:
                    self.probabilities[(s, a, s2)] = self.transitions[(s, a)][s2] / num_transitions

    def policy_iteration(self, env, gamma=0.99, theta=0.001, num_random_steps=10 ** 3, max_iterations=10 ** 3):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("Expected Discrete action space")

        self.estimate_transition_probabilities(env, num_iterations=num_random_steps)

        v = self.value_table
        policy = self.policy

        all_states = set([k[0] for k in self.probabilities.keys()]) | set([k[2] for k in self.probabilities.keys()])

        policy_stable = False
        i = 0
        while not policy_stable and i < max_iterations:
            logger.warning("Round {}".format(i))
            logger.debug("Policy Evaluation")
            # Policy evaluation
            delta = float("inf")
            while delta >= theta:
                delta = 0
                for s in all_states:
                    val = v[s]
                    v[s] = sum(
                        [self.probabilities[(s, policy[s], s2)] * (self.rewards[(s, policy[s], s2)] + gamma * v[s2]) for
                         s2
                         in all_states])
                    delta = max(delta, abs(val - v[s]))

            logger.warning("Delta was {}".format(delta))
            logger.debug("Policy Improvement")

            # Policy improvement
            policy_stable = True

            for s in all_states:
                old_action = policy[s]

                state_action_pairs = [(st, ac) for (st, ac) in self.transitions.keys() if st == s]
                best_action = None
                best_value = float("-inf")
                for _, a in state_action_pairs:
                    cur_val = 0.0
                    for s2 in self.transitions[(s, a)].keys():
                        cur_val += self.probabilities[(s, a, s2)] * (self.rewards[(s, a, s2)] + gamma * v[s2])

                    if cur_val > best_value:
                        best_value = cur_val
                        best_action = a
                if best_action is None:
                    logger.debug("No best action found for state {}".format(s))
                    best_action = 0
                policy[s] = best_action
                if best_action != old_action:
                    logger.debug("policy[{}] -> {}".format(s, best_action))
                    policy_stable = False
            i += 1

        self.policy = policy
        self.value_table = v

    def learn(self, env, algorithm=DPAlgorithm.POLICY_ITERATION, gamma=0.99, theta=0.01, num_random_steps=10 ** 3,
              max_iterations=10 ** 3):
        if algorithm == DPAlgorithm.POLICY_ITERATION:
            self.policy_iteration(env, gamma=gamma, theta=theta, num_random_steps=num_random_steps,
                                  max_iterations=max_iterations)
        elif algorithm == DPAlgorithm.VALUE_ITERATION:
            self.value_iteration(env, gamma=gamma, theta=theta, num_random_steps=num_random_steps)
        else:
            raise RuntimeError("Algorithm not implemented")

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

                action = self._select_action(env, state)
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

    def value_iteration(self, env, gamma=0.99, theta=0.01, num_random_steps=1000):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("Expected Discrete action space")

        self.estimate_transition_probabilities(env, num_iterations=num_random_steps)

        v = self.value_table
        policy = collections.defaultdict(int)

        all_states = set([k[0] for k in self.probabilities.keys()]) | set([k[2] for k in self.probabilities.keys()])
        # Policy evaluation
        delta = float("inf")
        while delta >= theta:
            delta = 0
            for s in all_states:
                state_action_pairs = [(st, ac) for (st, ac) in self.transitions.keys() if st == s]
                best_value = float("-inf")
                val = v[s]
                for _, a in state_action_pairs:
                    cur_val = 0.0
                    for s2 in self.transitions[(s, a)].values():
                        cur_val += self.probabilities[(s, a, s2)] * (self.rewards[(s, a, s2)] + gamma * v[s2])

                    if cur_val > best_value:
                        best_value = cur_val

                if best_value == float("-inf") :
                    logger.info("State {} has no best action".format(s))
                v[s] = best_value
                delta = max(delta, abs(val - v[s]))

        for s in all_states:

            state_action_pairs = [(st, ac) for (st, ac) in self.transitions.keys() if st == s]
            best_action = None
            best_value = float("-inf")
            for _, a in state_action_pairs:
                cur_val = 0.0
                for s2 in self.transitions[(s, a)].keys():
                    cur_val += self.probabilities[(s, a, s2)] * (self.rewards[(s, a, s2)] + gamma * v[s2])

                if cur_val > best_value:
                    best_value = cur_val
                    best_action = a
            if best_action is None:
                logger.info("No best action found for state {}".format(s))
                best_action = 0
            policy[s] = best_action

        self.policy = policy
        self.value_table = v


def main():
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "Taxi-v2"
    algorithm = DPAlgorithm.POLICY_ITERATION
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 10000
    agent = DPAgent()
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_random_steps = 1000
    num_test_episodes = 100
    while True:
        agent.learn(environment, algorithm, gamma=gamma, num_random_steps=num_random_steps)
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


if __name__ == "__main__":
    main()
