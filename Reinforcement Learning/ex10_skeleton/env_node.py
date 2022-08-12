import random

import numpy as np

from base_node import MonteCarloTreeSearchNode


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
        child_node = None
        # TODO: 1) a)
        a = self._untried_actions.pop()
        next_state, reward, done, _ = self.env.step_state(state=self.state, action=a)

        child_node = DeterministicResettableEnvMCTSNode(self.env, state=next_state, parent=self, is_terminal=done,
                                                        base_reward=reward)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.is_terminal

    def rollout(self):
        total_reward = 0
        current_state = self.state

        done = self.is_terminal
        while not done:
            random_action = random.choice(self.env.get_legal_actions(state=current_state))
            current_state, reward, done, _ = self.env.step_state(current_state, random_action)
            total_reward += reward

        return total_reward

    def backpropagate(self, result):
        self._total_children_reward += result
        self._number_of_visits += 1

        if self.parent:
            # only propagate result from rollout instead of self._total_children_reward
            # to parent as it already has the base reward from this node
            self.parent.backpropagate(result)
        pass

    def best_child(self, c_param=10):
        choice_scores = [((c.q / c.n) + (c_param * np.sqrt(2 * np.log(self.n) / c.n))) for c in self.children]
        best_choice = np.argmax(choice_scores)
        return self.children[best_choice]

    def render_to_string(self, show_children=True, uptolevel=3, indent=0) -> str:
        return_string = (" " * indent) + f"{self.state[0]}\n"

        return_string += (" " * indent) + f"{self.state[1]}"
        return_string += f"   q={(self.q):>6.1f}  | n={self.n:>6.0f} | q/n={(self.q / max(1, self.n)):>6.1f} | {'is_terminal' if self.is_terminal_node() else ''}\n"
        return_string += (" " * indent) + f"{self.state[2]}"

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
