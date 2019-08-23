import json
import numpy as np
import collections


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


class TabularPolicy(object):

    def __init__(self, default_value=None, random_defaults=None):
        self.random_defaults = random_defaults
        if random_defaults is not None:
            self.policy_table = {}
        elif default_value is None:
            self.policy_table = collections.defaultdict(int)
        else:
            self.policy_table = collections.defaultdict(lambda: default_value)

    def __call__(self, *args, **kwargs):
        arg = args[0]
        return self.__getitem__(arg)

    def __setitem__(self, key, value):
        self.policy_table[key] = value

    def __getitem__(self, item):
        if self.random_defaults is not None and item not in self.policy_table:
            self.policy_table[item] = self._random_sample()
        return self.policy_table[item]

    def _random_sample(self):
        return np.random.choice(self.random_defaults)

    @classmethod
    def greedy_from_q(cls, q_table: StateActionValueTable):
        policy = cls()
        for state in q_table.q.keys():
            action = q_table.get_best_action(state)
            policy[state] = action
        return policy

    @classmethod
    def sample_frozen_lake_policy(cls):
        policy = cls()
        policy.policy_table = {"0": 1, "1": 1, "2": 1, "3": 0, "4": 0, "5": 0, "6": 1, "7": 0, "8": 1, "9": 1, "10": 0,
                               "11": 0, "12": 0, "13": 0, "14": 2, "15": 0}
        return policy


class EpsilonGreedyTabularPolicy(object):

    def __init__(self, action_space_n, epsilon):
        if epsilon > 1 or epsilon <= 0:
            raise ValueError("Epsilon must be less or equal to 1 and bigger than 0.")
        if action_space_n <= 0:
            raise ValueError("Empty Action Space was given")
        self.action_space = range(action_space_n)
        self.epsilon = epsilon
        self.policy_table = {}

    def _initialize_state(self, state):
        initial_val = np.full(len(self.action_space), 1)
        initial_val = initial_val / initial_val.sum()
        self.policy_table[state] = initial_val

    def __call__(self, *args, **kwargs):
        arg = args[0]
        return self.__getitem__(arg)

    def __setitem__(self, key, value):
        new_val = np.full(len(self.action_space), self.epsilon / len(self.action_space))
        new_val[value] = 1 - self.epsilon + (self.epsilon / len(self.action_space))
        self.policy_table[key] = new_val

    def __getitem__(self, state):
        if state not in self.policy_table:
            self._initialize_state(state)
        val = self.policy_table[state]
        return np.random.choice(len(val), p=val)

    def get_probability(self, action, state):
        if state not in self.policy_table:
            self._initialize_state(state)
        return self.policy_table[state][action]

    @classmethod
    def random_policy(cls, action_space_n):
        return cls(action_space_n, 1.0)

    @classmethod
    def from_q(cls, action_space_n, epsilon, q_table: StateActionValueTable):
        policy = cls(action_space_n, epsilon)
        for state in q_table.q.keys():
            action = q_table.get_best_action(state)
            policy[state] = action
        return policy

    @classmethod
    def sample_frozen_lake_policy(cls, epsilon):
        policy_table = {"0": 1, "1": 1, "2": 1, "3": 0, "4": 0, "5": 0, "6": 1, "7": 0, "8": 1, "9": 1, "10": 0,
                        "11": 0, "12": 0, "13": 0, "14": 2, "15": 0}
        policy = cls(4, epsilon)
        for state in policy_table:
            policy[state] = policy_table[state]
        return policy


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


class CustomPolicy(object):

    def __init__(self, policy_func, probability_func):
        self.policy_func = policy_func
        self.probability_func = probability_func

    def __call__(self, *args, **kwargs):
        arg = args[0]
        return self.policy_func(arg)

    def get_probability(self, action, state):
        return self.probability_func(action, state)

    @classmethod
    def get_simple_blackjack_policy(cls):
        return cls(simple_blackjack_policy, simple_blackjack_probability)


def simple_blackjack_policy(state, limit=11):
    # We assume the state is a string form of a Blackjack-v0 state
    split = state.split(",")
    score = int(split[0][1:])
    dealer_score = int(split[1])
    usable_ace = split[2][1:-1] == "True"

    if score <= limit:
        return 1
    return 0


def simple_blackjack_probability(action, state, limit=11):
    if action == simple_blackjack_policy(state, limit=limit):
        return 1.0
    return 0.0


class EnvironmentNode(object):

    def __init__(self, id):
        self.id = id
        self.parents = set()
        self.children = set()


class EnvironmentModel(object):

    def __init__(self):
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.states = set()
        self.actions = None
        self.start_states = set()
        self.terminal_states = set()
        self.nodes = {}

    def random_action(self):
        return np.random.choice(self.actions)

    def random_state(self):
        return np.random.choice(len(self.states))

    def random_state_and_action(self):
        transitions = list(self.transitions.keys())
        return transitions[np.random.choice(len(transitions))]

    def estimate(self, env, exp_policy=None, num_iterations=100):
        """
        Performs a number of random actions and estimates the state-action-state probabilities depending on how often
        they occurred during this time.
        :param env: The environment to be evaluated
        :param exp_policy: An optional exploration policy to use
        :param num_iterations: how many steps to perform
        :return:
        """
        state = env.reset()
        state = str(state)
        self.actions = range(env.action_space.n)
        self.start_states.add(state)
        self.states.add(state)
        for _ in range(num_iterations):
            if exp_policy is not None:
                action = exp_policy(state)
            else:
                action = env.action_space.sample()
            new_state, reward, is_done, _ = env.step(action)
            new_state = str(new_state)
            self.append(state, action, new_state, reward, is_done)
            state = env.reset() if is_done else new_state
            state = str(state)
            if is_done:
                self.start_states.add(state)

    def get_probability(self, state, action, new_state):
        curr_trans = self.transitions[(state, action)]
        num_transitions = sum(curr_trans.values())
        if num_transitions == 0 or new_state not in curr_trans:
            return 0.0
        return curr_trans[new_state] / num_transitions

    def get_reward(self, state, action, new_state):
        return self.rewards[(state, action, new_state)]

    def sample(self, state, action):
        curr_trans = self.transitions[(state, action)]
        if len(curr_trans.keys()) == 0:
            return None

        other_states, counts = zip(*curr_trans.items())
        num_transitions = sum(counts)
        if num_transitions == 0:
            other_state = other_states[np.random.choice(len(other_states))]
            reward = self.rewards[(state, action, other_state)]
            return reward, other_state

        probs = np.array(counts) / num_transitions
        idx = np.random.choice(len(other_states), p=probs)

        other_state = other_states[idx]
        reward = self.rewards[(state, action, other_state)]
        return other_state, reward

    def append(self, state, action, new_state, reward, done=False):
        self.rewards[(state, action, new_state)] = reward
        self.transitions[(state, action)][new_state] += 1
        self.states.add(new_state)

        if state not in self.nodes.keys():
            self.nodes[state] = EnvironmentNode(state)

        if new_state not in self.nodes.keys():
            self.nodes[new_state] = EnvironmentNode(new_state)

        self.nodes[state].children.add(new_state)
        self.nodes[new_state].parents.add(state)

        if done:
            self.terminal_states.add(new_state)


