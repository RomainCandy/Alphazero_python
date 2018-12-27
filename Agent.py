import random

import numpy as np

import MCTS


class Agent:
    def __init__(self, name, num_simulations, action_size, c, model, training_loop=10):
        self.name = name
        self.num_simulations = num_simulations
        self.action_size = action_size
        self.c = c
        self.model = model
        self.mcts = None
        self.root = None
        self.training_loop = training_loop
        self.memo_predict = dict()

    def act(self, state, tau):
        if self.mcts is None:
            self.build_mcts(state)
        else:
            self.change_root(state)

        for ii in range(self.num_simulations):
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
            try:
                probs = probs[allowed_moves]
            except IndexError as err:
                import pdb
                pdb.set_trace()
                raise IndexError(err)
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
        if str(state.to_model()) in self.memo_predict:
            probs, value = self.memo_predict[str(state.to_model())]
        else:
            probs, value = self.model.predict(state.to_model())
            self.memo_predict[str(state.to_model())] = probs, value
            # print(allowed_moves, type(allowed_moves), allowed_moves.shape)
        if not allowed_moves.shape[0]:
            probs = np.ones_like(probs) / probs.shape[0]
            return value, probs, allowed_moves
        mask = np.ones(probs.shape, dtype=bool)
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
        if np.sum(pi*1.0) == 0:
            return pi, values
        pi = pi / np.sum(pi * 1.0)
        return pi, values

    @staticmethod
    def chose_action(pi, values, tau):
        if tau:
            action_idx = np.random.multinomial(1, pi)
            # print(action_idx, 'lol', pi)
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
        try:
            self.mcts.root = self.mcts.tree[state.id]
        except KeyError as err:
            import pdb
            pdb.set_trace()
            raise KeyError(err)

    def train(self, memory, batch_size=256):
        for _ in range(self.training_loop):
            batch = memory.sample(min(batch_size, len(memory)))
            self.model.train(batch)


class User:
    """
    A human player
    """
    def __init__(self, name, action_size):
        self.name = name
        self.action_size = action_size
        self.mcts = None

    def act(self, state, tau):
        assert tau == 0
        action = int(input('your move: '))
        while action not in state.action_possible:
            print('possible actions are {}'.format(state.action_possible))
            action = int(input('your move: '))
        pi = np.zeros(self.action_size)
        pi[action] = 1
        return action, pi, None, None

    @staticmethod
    def chose_action(pi, values, tau):
        if tau:
            action_idx = np.random.multinomial(1, pi)
            # print(action_idx, 'lol', pi)
            action = np.where(action_idx == 1)[0][0]
        else:
            best_actions = np.argwhere(pi == np.max(pi))
            action = random.choice(best_actions)[0]
        value = values[action]
        return action, value
