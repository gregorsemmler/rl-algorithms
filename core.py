import json
import numpy as np
import collections
import torch
from torch import nn, optim
import torch.nn.functional as F
from gym.spaces import box, discrete


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


class ApproximateValueFunction(object):

    def __init__(self, n_states, learning_rate, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.input_size = n_states
        self.hidden_size = self.input_size
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.x_batches = []
        self.y_batches = []

    def __call__(self, state):
        inp = self.state_to_network_input(state).to(self.device)
        output = self.model(inp)
        return float(output.detach().cpu().numpy())

    def state_to_network_input(self, state, dtype="torch.FloatTensor"):
        int_state = int(state)
        one_hot_encoded = np.zeros((1, self.input_size))
        one_hot_encoded[0, int_state] = 1
        return torch.from_numpy(one_hot_encoded).type(dtype)

    def append_x_y_pair(self, state, y):
        self.x_batches.append(self.state_to_network_input(state))
        self.y_batches.append(torch.FloatTensor([[y]]))

    def shuffle_batches(self):
        permutation = np.random.permutation(len(self.x_batches))
        self.x_batches = [self.x_batches[i] for i in permutation]
        self.y_batches = [self.y_batches[i] for i in permutation]

    def approximate(self, batch_size):
        self.model.train()
        losses = []

        while len(self.x_batches) > 0:
            x_batch = self.x_batches[:batch_size]
            y_batch = self.y_batches[:batch_size]

            self.x_batches = self.x_batches[batch_size:]
            self.y_batches = self.y_batches[batch_size:]

            x_batch = torch.cat(x_batch).to(self.device)
            y_batch = torch.cat(y_batch).to(self.device)

            self.model.zero_grad()

            out = self.model.forward(x_batch)
            loss = self.criterion(out, y_batch)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            self.optimizer.step()

        return losses


class ApproximateStateActionFunction(object):

    def __init__(self, n_states, n_actions, learning_rate, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.input_size = n_states + n_actions
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_size = self.n_states
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.x_batches = []
        self.y_batches = []

    def __call__(self, state, action):
        inp = self.state_action_to_network_input(state, action).to(self.device)
        output = self.model(inp)
        return float(output.detach().cpu().numpy())

    def state_action_to_network_input(self, state, action, dtype="torch.FloatTensor"):
        int_state = int(state)
        one_hot_s = np.zeros((1, self.n_states))
        one_hot_s[0, int_state] = 1
        int_action = int(action)
        one_hot_a = np.zeros((1, self.n_actions))
        one_hot_a[0, int_action] = 1
        inp = np.concatenate((one_hot_s, one_hot_a), axis=1)
        return torch.from_numpy(inp).type(dtype)

    def append_x_y_pair(self, state, action, y):
        self.x_batches.append(self.state_action_to_network_input(state, action))
        self.y_batches.append(torch.FloatTensor([[y]]))

    def shuffle_batches(self):
        permutation = np.random.permutation(len(self.x_batches))
        self.x_batches = [self.x_batches[i] for i in permutation]
        self.y_batches = [self.y_batches[i] for i in permutation]

    def approximate(self, batch_size):
        self.model.train()
        losses = []

        while len(self.x_batches) > 0:
            x_batch = self.x_batches[:batch_size]
            y_batch = self.y_batches[:batch_size]

            self.x_batches = self.x_batches[batch_size:]
            self.y_batches = self.y_batches[batch_size:]

            x_batch = torch.cat(x_batch).to(self.device)
            y_batch = torch.cat(y_batch).to(self.device)

            self.model.zero_grad()

            out = self.model.forward(x_batch)
            loss = self.criterion(out, y_batch)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            self.optimizer.step()

        return losses

    def get_q_max_pair(self, state):
        best_score = float("-inf")
        best_action = -1

        for action in range(self.n_actions):
            out = self.__call__(state, action)

            if out > best_score:
                best_score = out
                best_action = action
        return best_action, best_score

    def get_best_action(self, state):
        return self.get_q_max_pair(state)[0]

    def get_q_max(self, state):
        return self.get_q_max_pair(state)[1]


class ApproximatePolicy(object):

    def __init__(self, observation_space, n_actions, learning_rate, hidden_size=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        if isinstance(observation_space, box.Box):
            self.discrete_state_space = False
        elif isinstance(observation_space, discrete.Discrete):
            self.discrete_state_space = True
        else:
            raise ValueError("Unsupported Observation Space")
        self.input_size = observation_space.n if self.discrete_state_space else observation_space.shape[0]
        self.n_actions = n_actions
        self.hidden_size = hidden_size
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.n_actions)).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.state_batches = []
        self.action_batches = []
        self.value_batches = []

    def __call__(self, state):
        action_probs = self.get_probabilities(state)
        action = np.random.choice(self.n_actions, p=action_probs)
        return action

    def get_probabilities(self, state):
        inp = self.state_to_network_input(state).to(self.device)
        probs = self.model(inp)
        probs = F.softmax(probs, dim=-1)
        return probs.detach().cpu().numpy().squeeze()

    def state_to_network_input(self, state, dtype="torch.FloatTensor"):
        if self.discrete_state_space:
            int_state = int(state)
            state_np = np.zeros((1, self.input_size))
            state_np[0, int_state] = 1
        else:
            state_np = state.reshape(1, -1)
        return torch.from_numpy(state_np).type(dtype)

    def append(self, state, action, value):
        self.state_batches.append(self.state_to_network_input(state))
        self.action_batches.append(action)
        self.value_batches.append(torch.FloatTensor([value]))

    def shuffle_batches(self):
        permutation = np.random.permutation(len(self.state_batches))
        self.state_batches = [self.state_batches[i] for i in permutation]
        self.action_batches = [self.action_batches[i] for i in permutation]
        self.value_batches = [self.value_batches[i] for i in permutation]

    def policy_gradient_approximation(self, batch_size):
        self.model.train()
        losses = []

        while len(self.state_batches) > 0:
            state_batch = self.state_batches[:batch_size]
            action_batch = self.action_batches[:batch_size]
            value_batch = self.value_batches[:batch_size]

            self.state_batches = self.state_batches[batch_size:]
            self.action_batches = self.action_batches[batch_size:]
            self.value_batches = self.value_batches[batch_size:]

            state_batch = torch.cat(state_batch).to(self.device)
            value_batch = torch.cat(value_batch).to(self.device)

            self.model.zero_grad()

            out = self.model.forward(state_batch)
            action_log_prob = F.log_softmax(out, dim=1)
            loss = value_batch * action_log_prob[range(len(action_log_prob)), action_batch]
            loss = -loss.mean()
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            self.optimizer.step()

        return losses


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
        self.out_edges = set()
        self.in_edges = set()

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.id}"

    def __hash__(self):
        return hash(self.id)

    def add_child(self, action, new_state):
        self.children.add(new_state)
        self.out_edges.add((action, new_state))

    def add_parent(self, state, action):
        self.parents.add(state)
        self.in_edges.add((state, action))


class EnvironmentModel(object):

    def __init__(self):
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.actions = None
        self.start_states = set()
        self.terminal_states = set()
        self.nodes = {}

    def get_states(self):
        return set(self.nodes.keys())

    def get_node(self, state):
        return self.nodes[state]

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

        if state not in self.nodes.keys():
            self.nodes[state] = EnvironmentNode(state)

        if new_state not in self.nodes.keys():
            self.nodes[new_state] = EnvironmentNode(new_state)

        node = self.nodes[state]
        new_node = self.nodes[new_state]

        node.add_child(action, new_state)
        new_node.add_parent(state, action)

        if done:
            self.terminal_states.add(new_state)


