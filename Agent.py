import MCTS
import numpy as np
import random


class Agent:
    def __init__(self, name, num_simulations, action_size, c, model):
        self.name = name
        self.num_simulations = num_simulations
        self.action_size = action_size
        self.c = c
        self.model = model
        self.mcts = None
        self.root = None

    def act(self, state, tau):
        if self.mcts is None:
            self.build_mcts(state)
        else:
            self.change_root(state)

        for _ in range(self.num_simulations):
            self.simulate()
        pi, values = self.get_action_value()
        action, value = self.chose_action(pi, values, tau)
        next_state, _, _ = state.take_action(action)
        model_value = -1 * self.get_preds(next_state)[0]
        return action, pi, value, model_value

    def simulate(self):
        leaf_node, value, done, parents = self.mcts.move_to_leaf()
        value, parents = self.evaluate_leaf(leaf_node, value, done, parents)
        self.mcts.backward(leaf_node, value, parents)

    def evaluate_leaf(self, leaf_node, value, done, parent):
        if not done:
            value, probs, allowed_moves = self.get_preds(leaf_node.state)
            probs = probs[allowed_moves]
            for idx, move in enumerate(allowed_moves):
                new_state, _, _ = leaf_node.state.take_action(move)
                if new_state.id in self.mcts.tree:
                    node = self.mcts.tree[new_state.id]
                else:
                    node = MCTS.Node(new_state)
                    self.mcts.add_node(node)

                new_edge = MCTS.Edges(leaf_node, node, probs[idx], move)
                leaf_node.edges.append((move, new_edge))
        return value, parent

    def get_preds(self, state):
        allowed_moves = state.action_possible
        probs, value = self.model.predict(state.to_model())
        mask = np.zeros(probs.shape, dtype=bool)
        mask[allowed_moves] = False
        probs[mask] = -np.Inf
        probs = np.exp(probs) / np.sum(np.exp(probs), axis=0)
        return value, probs, allowed_moves

    def get_action_value(self):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float)
        for action, edge in edges:
            pi[action] += edge.stats['N']
            values[action] = edge.stats['Q']
        pi = pi / np.sum(pi * 1.0)
        return pi, values

    @staticmethod
    def chose_action(pi, values, tau):
        if tau:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]
        else:
            best_actions = np.argwhere(pi == np.max(pi))
            action = random.choice(best_actions)[0]
        value = values[action]
        return action, value

    def build_mcts(self, state):
        self.root = MCTS.Node(state)
        self.mcts = MCTS.MCTS(self.root, self.c)

    def change_root(self, state):
        self.mcts.root = self.mcts.tree[state.id]
