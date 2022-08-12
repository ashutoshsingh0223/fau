
import time

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : mctspy.tree.nodes.MonteCarloTreeSearchNode
        """
        self.root = node

    def best_action(self, total_simulation_seconds):
        """

        Parameters
        ----------
        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------
        best_node : MCTSNode
            The best child of the root node, according to MCTS
        """
        end_time = time.time() + total_simulation_seconds
        while time.time() < end_time:
            # Choose the node to run a rollout from
            # TODO: 2)
            current_node = self.root
            while not current_node.is_terminal_node():
                if not current_node.is_fully_expanded():
                    current_node = current_node.expand()
                    break
                else:
                    current_node = current_node.best_child()
            # TODO END

            reward = current_node.rollout()
            current_node.backpropagate(reward)
       
        return self.root.best_child(c_param=0.)


if __name__ == "__main__":
    from env_node import DeterministicResettableEnvMCTSNode
    import numpy as np
    from tictactoe import TicTacToeEnv    

    # Set up start state for MCTS
    env = TicTacToeEnv()
    obs = env.reset()
    initial_state = env.get_state()
    root = DeterministicResettableEnvMCTSNode(state=initial_state, env=env)
    mcts = MonteCarloTreeSearch(root)

    # Run MCTS
    best_node = mcts.best_action(total_simulation_seconds=10)

    print(root.render_to_string(uptolevel=2))
    print(best_node.render_to_string(show_children=False))