import collections
from enum import Enum

import gym
import logging
from gym import envs
from torch.utils.tensorboard import SummaryWriter

from core import TabularPolicy, EpisodeResult, EnvironmentModel, EpsilonGreedyTabularPolicy

logger = logging.getLogger(__file__)


class DPAlgorithm(Enum):
    POLICY_ITERATION = "POLICY_ITERATION"
    VALUE_ITERATION = "VALUE_ITERATION"


class DPAgent(object):

    def __init__(self):
        self.model = EnvironmentModel()
        self.policy = None
        self.v_table = None

    def policy_iteration(self, env, policy=None, v=None, b=None, gamma=0.99, theta=0.001, num_exploration_steps=1000,
                         max_iterations=1000):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("Expected Discrete action space")

        if policy is None:
            self.policy = TabularPolicy(random_defaults=env.action_space.n)
        else:
            self.policy = policy

        if v is None:
            self.v_table = collections.defaultdict(float)
        else:
            self.v_table = v

        self.model.estimate(env, exp_policy=b, num_iterations=num_exploration_steps)

        all_states = sorted(self.model.get_states())

        policy_stable = False
        i = 0
        while not policy_stable and i < max_iterations:
            # Policy evaluation
            delta = float("inf")
            while delta >= theta:
                delta = 0.0
                for s in all_states:
                    old_val = self.v_table[s]
                    action = self.policy(s)
                    new_val = sum([self.model.get_probability(s, action, s2) * (
                            self.model.get_reward(s, action, s2) + gamma * self.v_table[s2]) for s2 in all_states])
                    self.v_table[s] = new_val
                    delta = max(delta, abs(old_val - new_val))

            # Policy improvement
            policy_stable = True

            for s in all_states:
                old_action = self.policy(s)

                best_action = None
                best_value = float("-inf")
                for _, a in [(st, ac) for (st, ac) in self.model.transitions.keys() if st == s]:
                    cur_val = 0.0
                    for s2 in self.model.transitions[(s, a)].keys():
                        cur_val += self.model.get_probability(s, a, s2) * (
                                    self.model.get_reward(s, a, s2) + gamma * self.v_table[s2])

                    if cur_val > best_value:
                        best_value = cur_val
                        best_action = a
                if best_action is None:
                    logger.debug("No best action found for state {}".format(s))
                    best_action = 0
                self.policy[s] = best_action
                if best_action is not None and best_action != old_action:
                    logger.debug("policy[{}] -> {}".format(s, best_action))
                    policy_stable = False
            i += 1

            if i % 100 == 0:
                print("{} iterations done".format(i))

    def value_iteration(self, env, policy=None, v=None, b=None, gamma=0.99, theta=0.001, num_exploration_steps=1000):
        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise RuntimeError("Expected Discrete action space")

        if policy is None:
            self.policy = TabularPolicy(random_defaults=env.action_space.n)
        else:
            self.policy = policy

        if v is None:
            self.v_table = collections.defaultdict(float)
        else:
            self.v_table = v

        self.model.estimate(env, exp_policy=b, num_iterations=num_exploration_steps)
        all_states = sorted(self.model.get_states())

        # Policy evaluation
        delta = float("inf")
        while delta >= theta:
            delta = 0.0
            for s in all_states:
                old_val = self.v_table[s]
                action = self.policy(s)
                new_val = sum([self.model.get_probability(s, action, s2) * (
                            self.model.get_reward(s, action, s2) + gamma * self.v_table[s2]) for s2 in all_states])
                self.v_table[s] = new_val
                delta = max(delta, abs(old_val - new_val))

        for s in all_states:
            best_action = None
            best_value = float("-inf")
            for _, a in [(st, ac) for (st, ac) in self.model.transitions.keys() if st == s]:
                cur_val = 0.0
                for s2 in self.model.transitions[(s, a)].keys():
                    cur_val += self.model.get_probability(s, a, s2) * (
                                self.model.get_reward(s, a, s2) + gamma * self.v_table[s2])

                if cur_val > best_value:
                    best_value = cur_val
                    best_action = a
            if best_action is None:
                logger.debug("No best action found for state {}".format(s))
                best_action = 0
            self.policy[s] = best_action

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

    def predict(self, env, policy, algorithm, num_iterations, gamma=0.99, theta=0.01, num_random_steps=1000, v=None,
                b=None):
        if algorithm == DPAlgorithm.POLICY_ITERATION:
            self.policy_iteration(env, policy, b=b, v=v, gamma=gamma, theta=theta,
                                  num_exploration_steps=num_random_steps, max_iterations=num_iterations)
        elif algorithm == DPAlgorithm.VALUE_ITERATION:
            self.value_iteration(env, policy, b=b, v=v, gamma=gamma)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))

    def learn(self, env, policy, algorithm, num_iterations, gamma=0.99, theta=0.01, num_random_steps=1000, v=None,
              b=None):
        if algorithm == DPAlgorithm.POLICY_ITERATION:
            self.policy_iteration(env, policy, b=b, v=v, gamma=gamma, theta=theta,
                                  num_exploration_steps=num_random_steps, max_iterations=num_iterations)
        elif algorithm == DPAlgorithm.VALUE_ITERATION:
            self.value_iteration(env, policy, b=b, v=v, gamma=gamma)
        else:
            raise ValueError("Unknown Prediction Algorithm: {}".format(algorithm))


def prediction():
    policy = TabularPolicy.sample_frozen_lake_policy()
    env_name = "FrozenLake-v0"
    algorithm = DPAlgorithm.VALUE_ITERATION
    environment = gym.make(env_name)

    gamma = 0.99
    agent = DPAgent()
    num_iterations = 10000
    agent.predict(environment, policy, algorithm, b=policy, gamma=gamma, num_iterations=num_iterations)

    print("")
    pass


def control():
    estimation_policy = EpsilonGreedyTabularPolicy.sample_frozen_lake_policy(0.1)
    env_names = sorted(envs.registry.env_specs.keys())
    env_name = "FrozenLake-v0"
    algorithm = DPAlgorithm.POLICY_ITERATION
    env_spec = envs.registry.env_specs[env_name]
    environment = gym.make(env_name)
    test_env = gym.make(env_name)

    k = 0
    goal_returns = env_spec.reward_threshold
    gamma = 0.99

    writer = SummaryWriter(comment="-{}-{}".format(env_name, algorithm))

    max_rounds = 1000
    agent = DPAgent()
    test_best_result, test_best_return = None, float("-inf")
    test_returns = []
    num_iterations = 1000
    num_test_episodes = 100
    while True:
        agent.learn(environment, estimation_policy, algorithm, b=estimation_policy, gamma=gamma, num_iterations=num_iterations)
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
