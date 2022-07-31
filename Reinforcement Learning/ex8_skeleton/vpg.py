import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym

import math

from utils import episode_reward_plot


def compute_returns(rewards, next_value, discount):
    """ Compute returns based on episode rewards.

    Parameters
    ----------
    rewards : list of float
        Episode rewards.
    next_value : float
        Value of state the episode ended in. Should be 0.0 for terminal state, bootstrapped value otherwise.
    discount : float
        Discount factor.

    Returns
    -------
    list of float
        Episode returns.
    """
    returns = []
    ret = 0.0
    for value in reversed(rewards):
        ret = value + ret * discount
        returns.append(ret)
    return returns[::-1]


class TransitionMemory():
    """Datastructure to store episode transitions and perform return/advantage/generalized advantage calculations (GAE) at the end of an episode."""

    def __init__(self, gamma):
        self.gamma = gamma

        self.obs_list = []
        self.action_list = []
        self.rewards_list = []
        self.logp_list = []

        self.returns_list = []

        self.trajectory_start = 0

    def put(self, obs, action, reward, logp):
        """Put a transition into the memory."""
        self.obs_list.append(obs)
        self.action_list.append(action)
        self.rewards_list.append(reward)
        self.logp_list.append(logp)

    def get(self):
        """Get all stored transition attributes in the form of lists."""
        return self.obs_list, self.action_list, self.rewards_list, self.logp_list, self.returns_list

    def clear(self):
        """Reset the transition memory."""
        self.obs_list = []
        self.action_list = []
        self.rewards_list = []
        self.logp_list = []

        self.returns_list = []

        self.trajectory_start = 0

    def finish_trajectory(self, next_value):
        """Call on end of an episode. Will perform episode return or advantage or generalized advantage estimation (later exercise).
        
        Parameters
        ----------
        next_value:
            The value of the state the episode ended in. Should be 0.0 for terminal state.
        """

        traj_returns_list = compute_returns(rewards=self.rewards_list[self.trajectory_start: ], next_value=next_value,
                                            discount=self.gamma)
        self.returns_list.extend(traj_returns_list)
        self.trajectory_start = len(self.obs_list)


class ActorNetwork(nn.Module):
    """Neural Network used to learn the policy."""

    def __init__(self, num_observations, num_actions):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, obs):
        return self.net(obs)


class VPG():
    """The vanilla policy gradient (VPG) approach."""

    def __init__(self, env, episodes_update=5, gamma=0.99, lr=0.01):
        """ Constructor.
        
        Parameters
        ----------
        env : gym.Environment
            The object of the gym environment the agent should be trained in.
        episodes_update : int
            Number episodes to collect for every optimization step.
        gamma : float, optional
            Discount factor.
        lr : float, optional
            Learning rate used for actor and critic Adam optimizer.
        """

        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continous actions not implemented!')
        
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.memory = TransitionMemory(gamma)
        self.episodes_update = episodes_update

        self.actor_net = ActorNetwork(self.obs_dim, self.act_dim)
        self.optim_actor = optim.Adam(self.actor_net.parameters(), lr=lr)

    def learn(self, total_timesteps):
        """Train the VPG agent.
        
        Parameters
        ----------
        total_timesteps : int
            Number of timesteps to train the agent for.
        """
        obs = self.env.reset()

        # For plotting
        overall_rewards = []
        episode_rewards = []

        num_episodes = 0

        for timestep in range(1, total_timesteps + 1):
            # Do one step
            action, logp = self.predict(obs, train_returns=True)
            next_obs, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)

            # TODO put into transition buffer - DONE
            self.memory.put(obs, action, reward, logp)
            # Update current obs
            obs = next_obs

            if done:
                # TODO reset environment - DONE
                obs = self.env.reset()
                overall_rewards.append(sum(episode_rewards))
                episode_rewards = []

                # TODO finish trajectory - DONE
                # nect_value=0.0 as the state episode ended in was terminal, ensured by `if done` statement
                self.memory.finish_trajectory(next_value=0.0)

                num_episodes += 1

                if num_episodes == self.episodes_update:
                    # TODO optimize the actor - DONE
                    obs_list, action_list, reward_list, logp_list, returns_list = self.memory.get()
                    loss = self.calc_actor_loss(logp_list, returns_list)

                    self.optim_actor.zero_grad()
                    loss.backward()
                    self.optim_actor.step()

                    self.memory.clear()
                    num_episodes = 0

            # Episode reward plot
            if timestep % 500 == 0:
                episode_reward_plot(overall_rewards, timestep, window_size=5, step_size=1)

    def calc_actor_loss(self, logp_lst, return_lst):
        """Calculate actor "loss" for one batch of transitions."""
        print(len(logp_lst), len(return_lst))
        logp_lst = torch.stack(logp_lst)
        return_lst = torch.tensor(return_lst)

        loss = -logp_lst * return_lst
        return loss.mean()

    def predict(self, obs, train_returns=False):
        """Sample the agents action based on a given observation.
        
        Parameters
        ----------
        obs : numpy.array
            Observation returned by gym environment
        train_returns : bool, optional
            Set to True to get log probability of decided action and predicted value of obs.
        """

        # TODO

        obs = torch.FloatTensor(obs)

        logits_for_actions = self.actor_net(obs)
        distribution = Categorical(logits=logits_for_actions)

        action = distribution.sample()
        if train_returns:
            logp = distribution.log_prob(action)
            return action.item(), logp
        else:
            return action.item()


if __name__ == '__main__':
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    vpg = VPG(env)
    vpg.learn(100000)