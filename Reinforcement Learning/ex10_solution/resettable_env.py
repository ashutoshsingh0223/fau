from abc import ABC, abstractmethod
import gym
from gym.spaces import Discrete


class ResettableEnv(gym.Env, ABC):

    @abstractmethod
    def get_state(self):
        """Should return the environment state in a format expected by `set_state`."""
        pass

    @abstractmethod
    def set_state(self, new_state):
        """Should reset the environment state to the given state."""
        pass

    def get_legal_actions(self, state):
        assert isinstance(self.action_space, Discrete), "Legal actions can only be found for a discrete action space!"
        return list(range(self.action_space.n))

    def step_state(self, state, action):
        """
        Takes a single step in the environment with the specified state,
        returns everything that `step` does, and resets the environment back to its state at the end.
        """
        old_state = self.get_state()

        self.set_state(state.copy())
        return_values = self.step(action)
        self.set_state(old_state)

        return return_values
