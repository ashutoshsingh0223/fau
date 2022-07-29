import copy

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


class GridWorldEnv(gym.Env):
    """
    This environment is a variation on common grid world environments.
    Observation:
        Type: Box(2)
        Num     Observation            Min                     Max
        0       x position             0                       self.map.shape[0] - 1
        1       y position             0                       self.map.shape[1] - 1
    Actions:
        Type: Discrete(4)
        Num   Action
        0     Go up
        1     Go right
        2     Go down
        3     Go left
        Each action moves a single cell.
    Reward:
        Reward is 0 at non-goal points, 1 at goal positions, and -1 at traps
    Starting State:
        The agent is placed at [0, 0]
    Episode Termination:
        Agent has reached a goal or a trap.
    Solved Requirements:
        
    """

    metadata = {
        'render.modes': ['ascii'],
    }

    def __init__(self):
        self.map = [
            list("s   "),
            list("    "),
            list("    "),
            list("gt g"),
        ]

        self.map_ = np.array(self.map)

        self.low = 0
        self.y_high = self.map_.shape[1] - 1
        self.x_high = self.map_.shape[0] - 1

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([self.low, self.low]),
                                            high=np.array([self.x_high+1, self.y_high+1])
                                            )

        self.agent_position = [0, 0]
        self.current_observation = self.map_[self.agent_position[0]][self.agent_position[1]]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reward_done_from_current_observation(self):
        if self.current_observation == 'g':
            return 1, True
        elif self.current_observation == 't':
            return -1, True
        else:
            return 0, False

    def step(self, action):
        if action == 0:
            # Going up not possible at y low
            if self.agent_position[1] == self.low:
                pass
            else:
                self.agent_position[1] = self.agent_position[1] - 1

        elif action == 1:
            # Going right not possible at x high
            if self.agent_position[0] == self.x_high:
                pass
            else:
                self.agent_position[0] = self.agent_position[0] + 1

        elif action == 2:
            # Going down not possible at y high
            if self.agent_position[1] == self.y_high:
                pass
            else:
                self.agent_position[1] = self.agent_position[1] + 1

        else:
            # Going left not possible at x low
            if self.agent_position[0] == self.low:
                pass
            else:
                self.agent_position[0] = self.agent_position[0] - 1

        self.current_observation = self.map_[self.agent_position[0]][self.agent_position[1]]
        reward, done = self.reward_done_from_current_observation()

        return self.agent_position, reward, done, {}

    def reset(self):
        self.agent_position = [0, 0]
        self.current_observation = self.map_[self.agent_position[0]][self.agent_position[1]]
        return self.agent_position

    def render(self, mode='ascii'):
        rendered_map = copy.deepcopy(self.map)
        rendered_map[self.agent_position[0]][self.agent_position[1]] = "A"
        print("--------")
        for row in rendered_map:
            print("".join(["|", " "] + row + [" ", "|"]))
        print("--------")
        return None

    def close(self):
        pass
