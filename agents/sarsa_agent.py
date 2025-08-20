import numpy as np


class SARSA():

    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, alpha=0.5):
        self.state_n = 576
        self.action_n = action_space_n
        self.q_function = np.zeros((self.state_n, self.action_n))
        self.epsilon = 1.0
        self.gamma = gamma
        self.alpha = alpha


    def choose_action(self, state):
        q_values = self.q_function[state]
        policy = np.ones(self.action_n) * self.epsilon / self.action_n
        max_action = np.argmax(q_values)
        policy[max_action] += 1 - self.epsilon
        return np.random.choice(np.arange(self.action_n), p=policy)

    
    def learn(self,state, action, reward, next_state):
        next_action = self.choose_action(next_state)
        
        self.q_function[state][action] += self.alpha * (reward + self.gamma * self.q_function[next_state][next_action] - self.q_function[state][action])

    def update_epsilon(self, episode_number):
        self.epsilon = 1 / (episode_number + 1)
