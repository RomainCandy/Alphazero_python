import numpy as np
import sys
sys.path.append('..')
from GenericGame import GenericState


class GomokuGame:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.board = np.zeros((length, height), dtype=int)
        self.player_turn = 1
        self.state = StateGomoku(self.board, self.player_turn)
        self.action_size = length * height

    def reset(self):
        self.board = np.zeros((self.length, self.height), dtype=int)
        self.player_turn = 1
        self.state = StateGomoku(self.board, self.player_turn)
        return self.state

    def step(self, action):
        next_state, reward, done = self.state.take_action(action)
        self.state = next_state
        self.player_turn *= -1
        return next_state, reward, done

    def render(self, logger):
        logger.info(str(self.board))
        logger.info('-' * 50)

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return repr(self.state)


class StateGomoku(GenericState):
    def __init__(self, board, player_turn):
        super(StateGomoku, self).__init__(board, player_turn)

    def _get_corresp(self):
        return {1: 'X', -1: 'O', 0: '-'}

    def _get_actions(self):
        zz = np.argwhere(self.board == 0)
        return np.array([x + self.length * y for x, y in zz])

    def get_symmetries(self, pi):
        return [(self.to_model(), pi)]

    def take_action(self, action):
        # action un entier
        if action not in self.action_possible:
            raise ValueError(self)
        # column = self.board[:, action]
        new_board = np.array(self.board)
        i = action % self.length
        j = action // self.height
        new_board[i, j] = self.player_turn
        next_state = StateGomoku(new_board, -1 * self.player_turn)
        value = 0
        done = 0
        if not len(next_state.action_possible):
            done = 1
        if next_state.end_game(i, j):
            done = 1
            value = -1
        return next_state, value, done

    def end_game(self, i, j):
        if self._horizontal(j):
            return True
        elif self._vertical(i):
            return True
        elif self._diag_principale(i, j):
            return True
        elif self._diag_reverse(i, j):
            return True
        return False

    def is_terminal(self):
        for i in range(self.length):
            for j in range(self.height):
                if self.board[i, j] and self.end_game(i, j):
                    return True
        if not len(self.action_possible):
            return True
        return False

    def next_state(self):
        for action in self.action_possible:
            yield action, self.take_action(action)[0]

    def evaluate(self):
        return 0.1

    def _vertical(self, i):
        line = self.board[i, :]
        motif = ('+' + str(-1 * self.player_turn)) * 5
        line = "".join(['+' + str(x) for x in line])
        find = line.find(motif)
        return find != -1

    def _horizontal(self, j):
        column = self.board[:, j]
        motif = ('+' + str(-1 * self.player_turn)) * 5
        column = "".join(['+' + str(x) for x in column])
        find = column.find(motif)
        return find != -1

    def _diag_principale(self, i, j):
        diag = list()
        while j > 0 and i < self.length - 1:
            j -= 1
            i += 1
        while j <= self.height - 1 and i >= 0:
            diag.append(str(self.board[i, j]))
            j += 1
            i -= 1
        motif = ('+' + str(-1 * self.player_turn)) * 5
        diag = "".join(['+' + str(x) for x in diag])
        find = diag.find(motif)
        return find != -1 and diag[find - 1] != '-'

    def _diag_reverse(self, i, j):
        diag = list()
        while j > 0 and i > 0:
            j -= 1
            i -= 1
        while j < self.height and i < self.length:
            diag.append(str(self.board[i, j]))
            j += 1
            i += 1
        motif = ('+' + str(-1 * self.player_turn)) * 5
        diag = "".join(['+' + str(x) for x in diag])
        find = diag.find(motif)
        return find != -1 and diag[find - 1] != '-'

    def __str__(self):
        return '\n'.join([''.join([self.corresp[y] for y in x]) for x in self.board.tolist()])

    def __repr__(self):
        res = list()
        for i, row in enumerate(self.board):
            temp = list()
            for j, elem in enumerate(row):
                if elem == 0:
                    temp.append("'" + str(j*self.height + i) + "'")
                else:
                    temp.append(self.corresp[elem])
            res.append('\t'.join(temp))
        return '\n'.join(res)
