# -*- coding: utf-8 -*-
import gym


def simple_blackjack_policy(state):
    score, dealer_score, usable_ace = state

    if score <= 11:
        return 1
    return 0


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


def epsilon_greedy_tabular_sarsa(env, )


def main():
    blackjack = gym.make("Blackjack-v0")

    value_function = tabular_td0(blackjack, simple_blackjack_policy, alpha=1.0, gamma=1.0)

    print("")
    pass


if __name__ == "__main__":
    main()
