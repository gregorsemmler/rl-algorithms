# -*- coding: utf-8 -*-
import gym
import heapq


class StateActionValue(object):

    def __init__(self, state, action):
        self.state = state
        self.action = action

    def __hash__(self):
        return hash((self.state, self.action))

    def __eq__(self, other):
        return (self.state, self.action) == (other.state, other.action)

    def __ne__(self, other):
        return not(self == other)


class StateActionValueTable(object):

    def __init__(self, default_value=0.0, possible_actions=None):
        self.q = {}
        self.default_value = default_value
        self.possible_actions = possible_actions if possible_actions else []

    def _init_state_if_not_set(self, state, action=None):
        if state not in self.q:
            self.q[state] = {a: self.default_value for a in self.possible_actions}
            if action:
                self.q[state] = {action: self.default_value}

    def get_q_max(self, state):
        q_values = self.get_q_values_from_state(state)
        q_values = sorted(q_values.items(), key=lambda entry: -entry)  # Entry with highest value is first
        if len(q_values) == 0:
            return None
        return q_values[0]

    def get_q_values_from_state(self, state):
        self._init_state_if_not_set(state)
        return self.q[state]

    def get_q_value(self, state, action):
        self._init_state_if_not_set(state, action)
        return self.q[state][action]

    def set_q_value(self, state, action, value):
        self._init_state_if_not_set(state)
        self.q[state][action] = value


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


def epsilon_greedy_tabular_sarsa(env, epsilon=0.1, alpha=0.5, gamma=0.99, num_iterations=10**5):
    q = StateActionValueTable()

    i = 0
    while i < num_iterations:
        state = env.reset()
        action = env.action_space.sample()  # random first action
        done = False

        while not done:
            pass
            # TODO
            # action = policy(state)
            # new_state, reward, done, _ = env.step(action)
            #
            # if new_state not in v:
            #     v[new_state] = 0.0  # initialize
            #
            # v[state] = v[state] + alpha * (reward + gamma * v[new_state] - v[state])
        i += 1

    pass


def main():
    blackjack = gym.make("Blackjack-v0")

    value_function = tabular_td0(blackjack, simple_blackjack_policy, alpha=0.5, gamma=1.0, num_iterations=10 ** 5)

    print("")
    pass


if __name__ == "__main__":
    main()
