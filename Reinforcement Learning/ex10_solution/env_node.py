from base_node import MonteCarloTreeSearchNode

import numpy as np


class DeterministicResettableEnvMCTSNode(MonteCarloTreeSearchNode):
    """An MCTS-Node that has an environment and a state for the simulation."""

    def __init__(self, 
                    env,
                    state,
                    parent=None,
                    is_terminal=False,
                    base_reward=0):
        super().__init__(state.copy(), parent)
        self.env = env
        self._number_of_visits = 0.
        """In our case, the statistics is just the summed-up reward."""
        self._total_children_reward = base_reward
        self._untried_actions = None
        self.is_terminal = is_terminal

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.env.get_legal_actions(self.state)    
        return self._untried_actions

    @property
    def q(self):
        return self._total_children_reward

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        # TODO: 1) a)
        # Choose and remove one of the untried actions
        action = self.untried_actions.pop()

        # Find the successor state
        # Hint: you can use env.step_state(state, action) to simulate one round
        # (both players) of Tic-Tac-Toe.
        next_state, reward, done, _ = self.env.step_state(self.state, action)
        
        # Create a new child node for the search, and append it to children.
        child_node = DeterministicResettableEnvMCTSNode(
            self.env, next_state, parent=self, is_terminal=done, base_reward=reward
        )
        # TODO END

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.is_terminal

    def rollout(self):
        total_reward = 0
        # TODO: 3

        current_rollout_state = self.state
        done = self.is_terminal
        while not done:
            possible_moves = self.env.get_legal_actions(current_rollout_state)

            action = np.random.choice(possible_moves)

            current_rollout_state, reward, done, _ = (
                self.env.step_state(current_rollout_state, action)
            )

            total_reward += reward
        # TODO END
        return total_reward

    def backpropagate(self, result):
        # TODO: 4
        self._number_of_visits += 1.
        self._total_children_reward += result

        if self.parent:
            self.parent.backpropagate(result)
        
        # TODO END

    def best_child(self, c_param=10):
        # TODO: 1) b)
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]
        # TODO END


    def render_to_string(self, show_children=True, uptolevel=3, indent=0) -> str:
        return_string = (" " * indent) + f"{self.state[0]}\n"

        return_string +=  (" " * indent) + f"{self.state[1]}"
        return_string +=  f"   q={(self.q):>6.1f}  | n={self.n:>6.0f} | q/n={(self.q / max(1, self.n)):>6.1f} | {'is_terminal' if self.is_terminal_node() else ''}\n"
        return_string +=  (" " * indent) + f"{self.state[2]}"


        if show_children or uptolevel > 1:
            children = sorted(self.children, key=lambda c: (c.q / max(1, c.n)), reverse=True)
            for c in children[:2]:
                return_string += "\n" + c.render_to_string(
                    show_children=show_children,
                    uptolevel=uptolevel - 1,
                    indent=indent + 4)

        return return_string

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0