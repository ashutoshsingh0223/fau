import numpy as np
import random
from helper import action_value_plot, test_agent, visualize_rewards_vs_episodes

from gym_gridworld import GridWorldEnv

class SARSAQBaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.env = env

        # TODO: define a Q-function member variable self.Q
        # Remark: Use this kind of Q-value member variable for visualization tools to work, i.e. of shape [grid_height, grid_width, num_actions]
        # Q[y, x, z] is value of action z for grid position y, x
        self.Q = np.zeros([4, 4, 4], dtype=np.float32)

        self.sum_of_rewards = []
        self.episodes = []

    def action(self, s, epsilon=0.0):
        # TODO: implement epsilon-greedy action selection

        random_action_prob = np.random.random()
        if random_action_prob < epsilon:
            return self.env.action_space.sample()
        else:
            # Argmax not deterministic and may return different results for different runs if some actions are equiprobable
            # return np.argmax(self.Q[s[0], s[1], :])
            max_actions = np.argwhere(self.Q[s[0], s[1]] == np.amax(self.Q[s[0], s[1]])).flatten()
            return np.random.choice(max_actions)

class SARSAAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        for iter in range(n_timesteps):
            s = self.env.reset()
            a = self.action(s, epsilon=self.eps)

            reward_sum = -100
            while True:
                s_, r, done, info = self.env.step(a)
                a_ = self.action(s_, epsilon=self.eps)
                self.update_Q(s, a, r, s_, a_)
                reward_sum += r

                if done:
                    self.sum_of_rewards.append(np.mean(self.Q))
                    break
                else:
                    s = s_
                    a = a_

    def update_Q(self, s, a, r, s_, a_):
        discounted_value = self.g * self.Q[s_[0], s_[1], a_]
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (r + discounted_value - self.Q[s[0], s[1], a])


class QLearningAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=50000):
        # TODO: implement training loop

        for iter in range(n_timesteps):
            s = self.env.reset()
            reward_sum = -100
            while True:
                a = self.action(s, epsilon=self.eps)
                s_, r, done, info = self.env.step(a)
                self.update_Q(s, a, r, s_)
                reward_sum += r
                if done:
                    self.sum_of_rewards.append(np.mean(self.Q))
                    break
                else:
                    s = s_

    def update_Q(self, s, a, r, s_):
        # TODO: implement Q-value update rule
        discounted_value = self.g * np.max(self.Q[s_[0], s_[1], :])
        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (r + discounted_value - self.Q[s[0], s[1], a])


if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv(map_name='cliffwalking')

    discount_factor = 0.9
    learning_rate = 0.1
    epsilon = 0.4
    n_timesteps = 100000

    # Train SARSA agent
    sarsa_agent = SARSAAgent(env, discount_factor, learning_rate, epsilon)
    sarsa_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(sarsa_agent)
    # Uncomment to do a test run
    print("Testing SARSA agent...")
    test_agent(sarsa_agent, env, epsilon)

    sarsa_rewards = sarsa_agent.sum_of_rewards

    # Train Q-Learning agent
    qlearning_agent = QLearningAgent(env, discount_factor, learning_rate, epsilon)
    qlearning_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(qlearning_agent)
    # Uncomment to do a test run
    print("Testing Q-Learning agent...")
    test_agent(qlearning_agent, env, 0.0)

    q_learning_rewards = qlearning_agent.sum_of_rewards

    visualize_rewards_vs_episodes(q_learning_rewards=q_learning_rewards, sarsa_rewards=sarsa_rewards)
