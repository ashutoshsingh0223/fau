import numpy as np
import random
from helper import action_value_plot, test_agent

from gym_gridworld import GridWorldEnv


class SARSAQBaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.env = env

        # Get gridworld state space dimensionality
        self.grid_height = int(env.observation_space.high[0])
        self.grid_width = int(env.observation_space.high[1])

        # Get number of possible actions
        self.num_actions = env.action_space.n

        # Q[x, y, z] is value of action z for grid position x, y
        self.Q = np.zeros([self.grid_height, self.grid_width, self.num_actions], dtype=np.float32)

    def action(self, s, epsilon=0.0):
        rnd = np.random.uniform(0, 1)
        if rnd <= epsilon:
            action = np.random.randint(self.num_actions)
        else:
            max_q_actions = np.argwhere(self.Q[s[0], s[1]] == np.amax(self.Q[s[0], s[1]])).flatten()
            action = np.random.choice(max_q_actions)
        return action


class SARSAAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        t = 0
        while t <= n_timesteps:
            s = env.reset()
            a = self.action(s, self.eps)
            while True:
                s_, r, done, _ = env.step(a)
                a_ = self.action(s_, self.eps)
                t += 1

                self.update_Q(s, a, r, s_, a_)

                s = s_
                a = a_

                if done:
                    break

    def update_Q(self, s, a, r, s_, a_):
        # Q = Q + lr * (reward + gamma * Q' - Q)
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (
                    r + self.g * self.Q[s_[0], s_[1], a_] - self.Q[s[0], s[1], a])


class QLearningAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        t = 0
        while t <= n_timesteps:
            s = env.reset()
            while True:
                a = self.action(s, self.eps)
                s_, r, done, _ = env.step(a)
                t += 1

                self.update_Q(s, a, r, s_)

                s = s_

                if done:
                    break

    def update_Q(self, s, a, r, s_):
        # Q = Q + lr * (reward + gamma * max Q' - Q)
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (
                    r + self.g * np.max(self.Q[s_[0], s_[1]]) - self.Q[s[0], s[1], a])


if __name__ == "__main__":
    # Create environment
    #env = GridWorldEnv(map_name='standard')
    env = GridWorldEnv(map_name='cliffwalking')

    # Hyperparameters
    discount_factor = 0.98
    learning_rate = 0.05
    epsilon = 0.4
    n_timesteps = 100000

    # Train SARSA agent
    sarsa_agent = SARSAAgent(env, discount_factor, learning_rate, epsilon)
    sarsa_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(sarsa_agent)
    # print("Testing SARSA agent...")
    test_agent(sarsa_agent, env, epsilon)

    # Train Q-Learning agent
    qlearning_agent = QLearningAgent(env, discount_factor, learning_rate, epsilon)
    qlearning_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(qlearning_agent)
    # print("Testing Q-Learning agent...")
    test_agent(qlearning_agent, env, epsilon)
