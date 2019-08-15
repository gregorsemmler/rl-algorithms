import logging

from tensorboardX import SummaryWriter

import gym
from gym import envs

from tabular.dynamic_programming import DPAlgorithm, DPAgent

logger = logging.getLogger(__file__)


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
