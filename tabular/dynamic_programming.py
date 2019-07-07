# -*- coding: utf-8 -*-
import collections
import logging
from tensorboardX import SummaryWriter

import gym
from gym import envs

logger = logging.getLogger(__file__)


class DPAgent(object):

    def __init__(self):
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.probabilities = collections.defaultdict(float)
        pass

    # TODO
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

        print("")

    def policy_iteration(self, env, gamma=0.99, theta=0.01, num_random_steps=10 ** 3, max_iterations=10 ** 3):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("Expected Discrete action space")

        self.estimate_transition_probabilities(env, num_iterations=num_random_steps)

        v = collections.defaultdict(float)
        policy = collections.defaultdict(int)

        all_states = set([k[0] for k in self.probabilities.keys()]) | set([k[2] for k in self.probabilities.keys()])

        policy_stable = False
        i = 0
        while not policy_stable and i < max_iterations:
            logger.warning("Round {}".format(i))
            logger.warning("Policy Evaluation")
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
            logger.warning("Policy Improvement")

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
                    logger.warning("No best action found for state {}".format(s))
                    best_action = 0
                policy[s] = best_action
                if best_action != old_action:
                    logger.warning("p[{}] -> {}".format(s, best_action))
                    policy_stable = False

            i += 1

        return policy, v

    # TODO implement
    def value_iteration(self, env, gamma=0.99, theta=0.01, num_random_steps=1000):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("Expected Discrete action space")

        self.estimate_transition_probabilities(env, num_iterations=num_random_steps)

        v = collections.defaultdict(float)

        all_states = set([k(0) for k in self.transitions.keys()]) | set([k(2) for k in self.transitions.keys()])
        # Policy evaluation
        delta = float("inf")
        while delta >= theta:
            delta = 0
            for s in all_states:
                state_action_pairs = [(st, ac) for (st, ac) in self.transitions.keys() if st == s]
                best_action = None
                best_value = float("-inf")
                for _, a in state_action_pairs:
                    cur_val = 0.0
                    for s2 in self.transitions[(s, a)].values():
                        cur_val += self.probabilities[(s, a, s2)] * (self.rewards[(s, a, s2)] + gamma * v[s2])

                    if cur_val > best_value:
                        best_value = cur_val
                        best_action = a
                # TODO

                val = v[s]
                v[s] = sum(
                    [self.probabilities[(s, policy[s], s2)] * (self.rewards[(s, policy[s], s2)] + gamma * v[s2]) for s2
                     in all_states])
                delta = max(delta, abs(val - v[s]))

    # TODO
    # def q_value_iteration(self, env):
    #     for state in range(env.observation_space.n):
    #         for action in range(env.action_space.n):
    #             action_value = 0.0
    #             target_counts = self.transitions[(state, action)]
    #             total = sum(target_counts.values())
    #             for tgt_state, count in target_counts.items():
    #                 reward = self.rewards[(state, action, tgt_state)]
    #                 best_action = self.select_action(tgt_state)
    #                 action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
    #             self.values[(state, action)] = action_value


def main():
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "FrozenLake-v0"

    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)
    logger.info("test")

    agent = DPAgent()
    agent.policy_iteration(environment)
    print("")
    pass


if __name__ == "__main__":
    main()
