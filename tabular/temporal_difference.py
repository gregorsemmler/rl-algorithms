# -*- coding: utf-8 -*-
import gym
import numpy as np
import heapq


# TODO queue has PriorityQueues

class StateActionValue(object):

    def __init__(self, state, action):
        self.state = state
        self.action = action

    def __hash__(self):
        return hash((self.state, self.action))

    def __eq__(self, other):
        return (self.state, self.action) == (other.state, other.action)

    def __ne__(self, other):
        return not (self == other)


class StateActionValueTable(object):

    def __init__(self, default_value=0.0, possible_actions=()):
        self.q = {}
        self.default_value = default_value
        self.possible_actions = possible_actions

    def _init_state_if_not_set(self, state, action=None):
        if state not in self.q:
            self.q[state] = {a: self.default_value for a in self.possible_actions}
        if action:
            self.q[state] = {action: self.default_value}

    def __setitem__(self, key, value):
        if type(key) is not tuple or len(key) != 2:
            raise RuntimeError("Expected state-action pair as key")

        state, action = key
        self._init_state_if_not_set(state)
        self.q[state][action] = value

    def __getitem__(self, item):
        if type(item) is not tuple:
            # state supplied
            self._init_state_if_not_set(item)
            return self.q[item]
        if type(item) is tuple and len(item) == 2:
            state, action = item
            self._init_state_if_not_set(state, action)
            return self.q[state][action]
        raise RuntimeError("Expected state or state-action pair as key")

    def get_q_max(self, state):
        q_values = self.__getitem__(state)
        q_values = sorted(q_values.items(), key=lambda entry: -entry[1])  # Entry with highest value is first
        if len(q_values) == 0:
            return None
        return q_values[0]


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


def epsilon_greedy_from_q(env: gym.Env, q: StateActionValueTable, state, epsilon):
    if np.random.uniform(1) <= epsilon:
        return env.action_space.sample()
    else:
        action = q.get_q_max(state)

        if action is None:
            action = env.action_space.sample()
    return action


def epsilon_greedy_tabular_sarsa(env, epsilon=0.1, alpha=0.5, gamma=0.99, num_iterations=10 ** 5):
    q = StateActionValueTable()

    i = 0
    while i < num_iterations:
        state = env.reset()
        action = epsilon_greedy_from_q(env, q, state, epsilon)
        done = False

        while not done:
            new_state, reward, done, info = env.step(action)

            new_action = epsilon_greedy_from_q(env, q, new_state, epsilon)

            td_error = (reward + gamma * q.get(new_state, new_action) - q.get(state, action))
            new_q_val = q.get(state, action) + alpha * td_error
            q.set(state, action, new_q_val)

            state = new_state
            action = new_action
        i += 1

    pass


def main():
    blackjack = gym.make("Blackjack-v0")

    q = StateActionValueTable()
    q[5, 2] = 5
    asdf = q[5]
    bdef = q[5,2]
    xyz = q[3,4]

    value_function = tabular_td0(blackjack, simple_blackjack_policy, alpha=0.5, gamma=1.0, num_iterations=10 ** 5)

    print("")
    pass


if __name__ == "__main__":
    main()
