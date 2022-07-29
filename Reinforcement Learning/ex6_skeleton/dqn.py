import random
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from helper import visualize_agent, episode_reward_plot

from collections import deque


class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.capacity = capacity
        self.buffer = deque()

    def put(self, obs, action, reward, next_obs, done):
        """Put a tuple of (obs, action, rewards, next_obs, done) into the replay buffer.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        self.buffer.append((obs, action, reward, next_obs, done,))

        if len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer."""
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)
    
    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super(DQNNetwork, self).__init__()
        self.linear = nn.Linear(in_features=num_obs, out_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(in_features=64, out_features=num_actions)
        
    def forward(self, x):
        return self.out(self.relu(self.linear(x)))


class DQN():
    """The DQN method."""

    def __init__(self, env, replay_size=10000, batch_size=32, gamma=0.99, sync_after=5, lr=0.001):
        """ Initializes the DQN method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        sync_after: int
            Timesteps after which the target network should be synchronized with the main network.
        lr: float
            Adam optimizer learning rate.        
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize DQN network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        # TODO: Initialize DQN target network, load parameters from DQN network
        self.target_dqn_net = DQNNetwork(self.obs_dim, self.act_dim)

        self.target_dqn_net.load_state_dict(self.dqn_net.state_dict())

        # Set up optimizer, only needed for DQN network
        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)

    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []

        obs = self.env.reset()
        for timestep in range(1, timesteps + 1):
            epsilon = epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)
            
            next_obs, reward, done, _ = env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, done)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if done:
                obs = env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                
            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_msbe_loss()

                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            if timestep % self.sync_after == 0:
                self.target_dqn_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % 500 == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
    
    def predict(self, state, epsilon=0.0):
        """Predict the best action based on state. With probability epsilon take random action
        
        Returns
        -------
        int
            The action to be taken.
        """
        prob = np.random.random()
        if prob < epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state).unsqueeze(0)
            Q = self.dqn_net(state)
            action = torch.argmax(Q).item()
            return action

    def compute_msbe_loss(self):
        """Compute the MSBE loss between self.dqn_net predictions and expected Q-values.
        
        Returns
        -------
        float
            The MSE between Q-value prediction and expected Q-values.
        """
        obs, actions, rewards, next_obs, done = self.replay_buffer.get(self.batch_size)

        obs = torch.FloatTensor(torch.stack([torch.tensor(ob) for ob in obs]))
        actions = torch.LongTensor(actions)
        done = torch.FloatTensor(done)
        next_obs = torch.FloatTensor(torch.stack([torch.tensor(ob) for ob in next_obs]))

        rewards = torch.FloatTensor(rewards)

        q_values = self.dqn_net(obs)
        # select q_values for actions that were actually takes i.e `actions` to calculate loss
        # gather dim = 1 walk alon axis 1 i.e one row at a time(one q vector)
        # and select value of for action from actions
        # since q_values[batch, num_actions] have a batch dimention actions[batch, ]
        # need one as well added by adding one axis at dim=1 to get a shape[batch, 1], then
        # squeeze out extra axis to have a tensor of floats[batch, ]
        q_values = q_values.gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_dqn_net(next_obs)
        # select max over all actions. Returns values and indices, select values
        next_q_values = next_q_values.max(1)[0]

        update = rewards + self.gamma * (1 - done) * next_q_values

        return F.mse_loss(q_values, update, reduction='mean')


def epsilon_by_timestep(timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
    """Linearily decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps"""
    if timestep >= frames_decay:
        return epsilon_final
    else:
        epsilon = epsilon_start - ((epsilon_start - epsilon_final) * timestep / 10000)
        return epsilon


if __name__ == '__main__':
    # Create gym environment
    env_id = "CartPole-v1"
    env = gym.make(env_id)

    # Plot epsilon rate over time
    plt.plot([epsilon_by_timestep(i) for i in range(50000)])
    plt.show()

    # Train the DQN agent
    dqn = DQN(env)
    dqn.learn(30000)

    # Visualize the agent
    visualize_agent(env, dqn)