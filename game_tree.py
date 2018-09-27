from collections import deque


class Node:
    """ A node in the game tree.
    """
    def __init__(self, state, net, move=None, parent=None):
        """
         Q(s, a) = 1/N (s, a) s' |s,a→s 0 V (s'),
        Args:
            state: state of the game
            move: last played move
            parent: previous position
        """
        self.state = state
        self.move = move
        self.parent = parent
        self.prior = prior
        self.visits = 0
        self.wins = 0
        self.Q = 0
        self.untried_moves = state.get_moves()
        self.children = list()
        self.player = state.player

    def add_children(self, new_state, prior, move):
        """
        Remove move from untried_moves and add a new child node for this move.
            Return the added child node
        Args:
            move: move to try
            new_state: new state after playing tha move
            prior: prior given by the neural network to select each move given the state

        Returns:
            added child node
        """
        child = Node(state=new_state, prior=prior, move=move, parent=self)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def is_leaf(self):
        return not self.untried_moves

    def update(self, results):
        self.visits += 1
        self.wins += results

    def select_child(self):
        """
        a t = argmax Q(s t , a) + U (s t , a) , using a variant of the PUCT algorithm
        """
        pass

    def backup(self, v):
        self.visits += 1
        self.wins += v
        self.Q = self.wins / self.visits
        if self.parent:
            self.parent.backup(v)

    def __str__(self):
        queue = deque()
        queue.append(self)
        view = list()
        while queue:
            new_queue = deque()
            while queue:
                node = queue.popleft()
                view.append(str(node.state))
                for child in node.children:
                    new_queue.append(child)
            view.append('\n')
            queue = new_queue
        return ' '.join(view)


class Game:
    def __init__(self):
        self.state = 1
        self.player = 'yo'

    def get_moves(self):
        return self.state + 1, self.state + 2


if __name__ == '__main__':
    # pass
    node1 = Node(1)
    for i in range(2, 4):
        node1.add_children(i)
    for i, child in enumerate(node1.children, 1):
        for k in range(2, 4):
            child.add_children(k*i*'1')
    print(node1)
