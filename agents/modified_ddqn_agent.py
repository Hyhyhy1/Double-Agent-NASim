import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Базовый класс Agent
class Agent:
    def __init__(self, gamma=0.99, epsilon=1.0, batch_size=128, lr=0.001,
                 epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000000, is_learning=True):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.is_learning = is_learning
        self.memory = PrioritizedReplayBuffer(mem_size)

    def save(self, state, action, reward, new_state, done):
        self.memory.add((state, action, reward, new_state, done))

    def reduce_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)

# 2. Приоритетный буфер 
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []
            
        prios = self.priorities[:len(self.buffer)] if len(self.buffer) < self.capacity else self.priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

# 3. Нейронная сеть для агентов
class QNN(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, action_size)
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


class DoubleQAgent(Agent):
    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, epsilon=1.0, 
                 batch_size=128, lr=0.0003, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, tau=0.001, is_learning=True):
        
        super().__init__(gamma=gamma, epsilon=epsilon, batch_size=batch_size,
             lr=lr, epsilon_dec=epsilon_dec, epsilon_end=epsilon_end,
             mem_size=mem_size, is_learning=is_learning)

        self.action_space_n = action_space_n
        self.tau = tau

        self.q_func = QNN(observation_space_shape, action_space_n, 42).to(device)
        self.q_func_target = QNN(observation_space_shape, action_space_n, 42).to(device)
        self.optimizer = optim.Adam(self.q_func.parameters(), lr=lr, weight_decay=1e-5)
        
        self.soft_update(tau=1.0)


    def choose_action(self, state):
        if np.random.random() > self.epsilon or not self.is_learning:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            self.q_func.eval()
            with torch.no_grad():
                action_values = self.q_func(state)
            return torch.argmax(action_values).item()
        else:
            return np.random.randint(self.action_space_n)


    def soft_update(self, tau=None):
        tau = tau or self.tau
        for target_param, local_param in zip(self.q_func_target.parameters(), self.q_func.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def learn(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        transitions, indices, weights = self.memory.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(device)
        

        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        

        Q_targets_next = self.q_func_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.q_func(states).gather(1, actions)
        

        errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)
        

        loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
        

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_func.parameters(), 1.0)
        self.optimizer.step()
        

        self.soft_update()
        

        self.epsilon = max(self.epsilon_min, self.epsilon - (1.0 - self.epsilon_min)/10000)