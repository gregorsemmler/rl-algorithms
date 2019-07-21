# -*- coding: utf-8 -*-
import json
import collections


class TabularPolicy(object):

    def __init__(self, default_value=None):
        if default_value is None:
            self.policy_table = collections.defaultdict(int)
        else:
            self.policy_table = collections.defaultdict(lambda: default_value)

    def __call__(self, *args, **kwargs):
        arg = args[0]
        return self.__getitem__(arg)

    def __setitem__(self, key, value):
        self.policy_table[key] = value

    def __getitem__(self, item):
        return self.policy_table[item]


class StateActionValueTable(object):

    def __init__(self, default_value=0.0, possible_actions=()):
        self.q = {}
        self.default_value = default_value
        self.possible_actions = possible_actions

    def _init_state_if_not_set(self, state, action=None):
        if state not in self.q:
            self.q[state] = {a: self.default_value for a in self.possible_actions}
        if action not in self.q[state] and action is not None:
            self.q[state][action] = self.default_value

    def __setitem__(self, key, value):
        if type(key) is not tuple or len(key) != 2:
            raise RuntimeError("Expected state-action pair as key")
        state, action = key
        self._init_state_if_not_set(state)
        self.q[state][action] = value

    def __getitem__(self, item):
        if type(item) is not tuple:  # state was supplied
            self._init_state_if_not_set(item)
            return self.q[item]
        if type(item) is tuple and len(item) == 2:
            state, action = item
            self._init_state_if_not_set(state, action)
            return self.q[state][action]
        raise RuntimeError("Expected state or state-action pair as key")

    def get_q_max_pair(self, state):
        q_values = self.__getitem__(state)
        q_values = sorted(q_values.items(), key=lambda entry: -entry[1])  # Entry with highest value is first
        if len(q_values) == 0:
            return None
        return q_values[0]

    def get_q_max_pair_from_sum(self, state, q2):
        q_values = self.__getitem__(state)
        q_values2 = q2[state]
        keys1 = set(q_values.keys())
        keys2 = set(q_values2.keys())
        q_sum = {k: q_values[k] + q_values2[k] for k in keys1 & keys2}
        for k in keys1 - keys2:
            q_sum[k] = q_values[k]
        for k in keys2 - keys1:
            q_sum[k] = q_values2[k]
        q_sum_values = sorted(q_sum.items(), key=lambda entry: -entry[1])  # Entry with highest value is first
        return q_sum_values[0]

    def get_q_max(self, state):
        return self.get_q_max_pair(state)[1]

    def get_best_action(self, state, q2=None):
        if q2 is not None:
            return self.get_q_max_pair_from_sum(state, q2)[0]
        return self.get_q_max_pair(state)[0]

    def to_json_file(self, filename):
        with open(filename, "w") as f:
            f.write(json.dumps(self.__dict__))

    def from_json_file(self, filename):
        with open(filename, "r") as f:
            content = json.load(f)
        self.q = content["q"]
        self.default_value = content["default_value"]
        self.possible_actions = content["possible_actions"]


class EpisodeResult(object):

    def __init__(self, env, start_state):
        self.env = env
        self.states = [start_state]
        self.actions = []
        self.rewards = []

    def append(self, action, reward, state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def calculate_return(self, gamma):
        total_return = 0.0
        for k in range(len(self.rewards)):
            total_return += gamma ** k * self.rewards[k]
        return total_return