import numpy as np


class Node:

    def __init__(self, state):

        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        return not bool(self.edges)

    def __str__(self):
        return str(self.state) + ' ' + str(self.edges)

    def __repr__(self):
        return self.__str__()


class Edges:

    def __init__(self, state_in, state_out, prior, action):
        self.state_in = state_in
        self.state_out = state_out
        self.action = action
        self.player_turn = state_in.player_turn
        self.id = '{} -> {}'.format(state_in.id, state_out.id)
        self.stats = {'N': 0,
                      'W': 0,
                      'Q': 0,
                      'P': prior
                      }

    def __repr__(self):
        return 'in: ' + str(self.state_in) + ' out: ' + str(self.state_out)


class MCTS:

    def __init__(self, root, c):
        self.root = root
        self.tree = {}
        self.c = c
        self.add_node(root)

    def move_to_leaf(self):
        parents = []
        current_node = self.root
        done = 0
        value = 0
        while not current_node.is_leaf():
            best_qu = -np.Inf
            if current_node == self.root:
                epsilon = 0.25
                nu = np.random.dirichlet([0.03] * len(current_node.edges))
            else:
                epsilon = 0
                nu = np.zeros(len(current_node.edges))

            nb = 0
            for _, edge in current_node.edges:
                nb += edge.stats['N']

            for idx, (action, edge) in enumerate(current_node.edges):
                u = self.c*(((1 - epsilon) *
                            edge.stats['P'] + epsilon * nu[idx]) *
                            np.sqrt(nb) / (1 + edge.stats['N'])
                            )
                q = edge.stats['Q']
                if q + u > best_qu:
                    best_qu = q + u
                    best_action = action
                    best_edge = edge

            _new_state, value, done = current_node.state.take_action(best_action)
            current_node = best_edge.state_out
            parents.append(best_edge)

        return current_node, value, done, parents

    @staticmethod
    def backward(leaf, value, parents):
        current_player = leaf.state.player_turn
        for edge in parents:
            player_turn = edge.player_turn
            direction = 2 * (player_turn == current_player) - 1
            edge.stats['N'] += 1
            edge.stats['W'] += value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node):
        self.tree[node.id] = node

    def __len__(self):
        return len(self.tree)

    def __str__(self):
        return repr(self.tree)
