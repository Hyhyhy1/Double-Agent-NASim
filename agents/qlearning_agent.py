import numpy as np

class Q_learning():

    def __init__(self, observation_space_shape, action_space_n, gamma=0.99, alpha=0.1):
        self.state_n = 24576
        self.action_n = action_space_n
        self.q_function = np.zeros((self.state_n, self.action_n))
        self.epsilon = 0.1
        self.gamma = gamma
        self.alpha = alpha


    def choose_action(self, state):
        state = state.astype(int)
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(self.action_n))
        else:
            return np.argmax(self.q_function[state])

    
    def learn(self, state, action, reward, next_state):
        td_target = reward + self.gamma * self.q_function[next_state][np.argmax(self.q_function[next_state])]
        self.q_function[state][action] += self.alpha * (td_target - self.q_function[state][action])


    def update_epsilon(self, episode_number):
        self.epsilon = 1 / (episode_number + 1)
