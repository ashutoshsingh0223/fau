from gym import spaces
from gym.utils import seeding
import numpy as np
from resettable_env import ResettableEnv


def clamp(v, minimal_value, maximal_value):
    return min(max(v, minimal_value), maximal_value)


AGENT_IDX = 1
ADVERSARY_IDX = 2


class TicTacToeEnv(ResettableEnv):
    """ A simple tic-tac-toe environment.
    """

    metadata = {
        'render.modes': ['ascii'],
    }

    def __init__(self):
        """Game board. 0 is empty, 1 is the agent, 2 is the adversary."""
        self.board = np.zeros((3, 3), dtype=np.int32)
        self.action_space = spaces.Discrete(9)
        # Each observation is simply the map.
        self.observation_space = spaces.Box(0, 2, shape=(3, 3), dtype=np.float32)

    def step(self, action):
        action = (action // 3, action % 3)

        action_was_valid = False
        # Move the player
        if self.board[action] == 0:
            action_was_valid = True
            self.board[action] = AGENT_IDX

            # Adversary's move (only if the player did move.)
            empty_places = np.where(self.board == 0)  # (np.where returns a tuple.)
            empty_places = np.stack(empty_places).T
            # Another option how to implement deterministic environment
            # Seed np with the weighted sum of the board
            # np.random.seed(
            #     np.sum(self.board)
            # )
            if len(empty_places) > 0:
                adversary_action_index = 0  # np.random.choice(list(range(len(empty_places))))
                adversary_action = empty_places[adversary_action_index]
                self.board[adversary_action[0], adversary_action[1]] = ADVERSARY_IDX

        # Board is full
        done = False or np.all(self.board != 0)
        # Any row is entirely ones
        agent_wins = self._check_win(AGENT_IDX)
        adversary_wins = self._check_win(ADVERSARY_IDX) and not agent_wins

        done = done or agent_wins or adversary_wins

        reward = -0.01 + (
            1 if agent_wins else 0
        ) + (
                     -1 if adversary_wins else 0
                 ) + (
                     -1 if not action_was_valid else 0
                 )

        return np.array(self.board), reward, done, {}

    def _check_win(self, index):
        return np.any([
            (self.board[row_index] == index).all()
            for row_index in range(self.board.shape[0])
        ]) or np.any([
            (self.board[:, col_index] == index).all()
            for col_index in range(self.board.shape[1])
        ]) or (  # Diagonal
                       self.board[0, 0] == self.board[1, 1] == self.board[2, 2] == index
               ) or (  # Diagonal
                       self.board[0, 2] == self.board[1, 1] == self.board[2, 0] == index
               )

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int)
        return np.array(self.board)

    def render(self, mode='ascii'):
        print(self.board)

    def close(self):
        pass

    def get_state(self):
        return self.board

    def set_state(self, new_state):
        self.board = new_state
