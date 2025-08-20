import random
from pprint import pprint

import numpy as np


class ReplayMemory:
    """Experience Replay for Tabular Q-Learning agent """

    def __init__(self, capacity, s_dims):
        self.capacity = capacity
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int32)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [self.s_buf[sample_idxs],
                 self.a_buf[sample_idxs],
                 self.next_s_buf[sample_idxs],
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]
        return batch


class TabularQFunction:
    """Tabular Q-Function """

    def __init__(self, num_actions):
        self.q_func = dict()
        self.num_actions = num_actions

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = str(x.astype(np.int32))
        if x not in self.q_func:
            self.q_func[x] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_func[x]

    def forward_batch(self, x_batch):
        return np.asarray([self.forward(x) for x in x_batch])

    def update(self, s_batch, a_batch, delta_batch):
        for s, a, delta in zip(s_batch, a_batch, delta_batch):
            q_vals = self.forward(s)
            q_vals[a] += delta

    def get_action(self, x):
        return int(self.forward(x).argmax())

    def display(self):
        pprint(self.q_func)


class TabularQLearningAgent:
    """A Tabular. epsilon greedy Q-Learning Agent using Experience Replay """

    def __init__(self,
                 observation_space_shape, 
                 action_space_n,
                 seed=None,
                 lr=0.001,
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99):

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.num_actions = action_space_n
        self.obs_dim = observation_space_shape

        # Training related attributes
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(
            1.0, self.final_epsilon, self.exploration_steps
        )
        self.batch_size = batch_size
        self.discount = gamma
        self.steps_done = 0

        # Q-Function
        self.qfunc = TabularQFunction(self.num_actions)

        # replay setup
        self.replay = ReplayMemory(replay_size, self.obs_dim)


    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon


    def get_egreedy_action(self, o):
        epsilon = self.get_epsilon()
        if random.random() > epsilon:
            return self.qfunc.get_action(o)
        return random.randint(0, self.num_actions-1)


    def optimize(self):
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # get q_vals for each state and the action performed in that state
        q_vals_raw = self.qfunc.forward_batch(s_batch)
        q_vals = np.take_along_axis(q_vals_raw, a_batch, axis=1).squeeze()

        # get target q val = max val of next state
        target_q_val_raw = self.qfunc.forward_batch(next_s_batch)
        target_q_val = target_q_val_raw.max(axis=1)
        target = r_batch + self.discount*(1-d_batch)*target_q_val

        # calculate error and update
        td_error = target - q_vals
        td_delta = self.lr * td_error

        # optimize the model
        self.qfunc.update(s_batch, a_batch, td_delta)

        q_vals_max = q_vals_raw.max(axis=1)
        mean_v = q_vals_max.mean().item()
        mean_td_error = np.absolute(td_error).mean().item()
        return mean_td_error, mean_v
    

    def learn(self, obs, action, reward, next_obs, done):
        self.replay.store(obs, action, next_obs, reward, done)
        self.optimize()
        pass