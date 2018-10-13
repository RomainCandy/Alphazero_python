import numpy as np
import sys
sys.path.append('..')
from GenericGame import GenericState


class Connect4Game:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.board = np.zeros((length, height), dtype=int)
        self.player_turn = 1
        self.state = StateConnect4(self.board, self.player_turn)

    def reset(self):
        self.board = np.zeros((self.length, self.height), dtype=int)
        self.player_turn = 1
        self.state = StateConnect4(self.board, self.player_turn)
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


class StateConnect4(GenericState):
    def __init__(self, board, player_turn):
        super(StateConnect4, self).__init__(board, player_turn)
        # self.board = board
        # self.length, self.height = board.shape
        # self.action_possible = self._get_actions()
        # self.player_turn = player_turn
        # self.corresp = {1: 'X', -1: 'O', 0: '-'}
        # self.id = self.__str__()

    def take_action(self, action):
        # action un entier
        if action not in self.action_possible:
            print(action, self.action_possible)
            raise ValueError(self)
        # column = self.board[:, action]
        new_board = np.array(self.board)
        index = 'ROFL'
        for index, pawn in reversed(list(enumerate(new_board[:, action]))):
            if not pawn:
                new_board[index, action] = self.player_turn
                break
        next_state = StateConnect4(new_board, -1 * self.player_turn)
        value = 0
        done = 0
        if not len(next_state.action_possible):
            done = 1
        if next_state.connect4(index, action):
            done = 1
            value = -1
        return next_state, value, done

    def connect4(self, index, action):
        if self._horizontal(action):
            return True
        elif self._vertical(index):
            return True
        elif self._diag_principale(index, action):
            return True
        elif self._diag_reverse(index, action):
            return True
        return False

    def to_model(self):
        board = np.zeros((2, self.length, self.height))
        board[0] = self.board
        board[1] = self.player_turn
        return board

    def get_symmetries(self, pi):
        # THINK LATER ON HOW TO MAKE THIS WORK (problem with allowed_moves)

        # board = np.zeros((2, self.length, self.height))
        # board[0] = np.flip(self.board, 1)
        # board[1] = self.player_turn
        # return [(self.to_model(), pi), (board, pi[::-1])]
        return [(self.to_model(), pi)]

    def _get_corresp(self):
        return {1: 'X', -1: 'O', 0: '-'}

    def _get_actions(self):
        return np.where(self.board[0] == 0)[0]

    def _vertical(self, index):
        line = self.board[index, :]
        # print(line, index, 'lol')
        motif = ('+' + str(-1 * self.player_turn)) * 4
        line = "".join(['+' + str(x) for x in line])
        # print('motif ', motif, 'line ', line)
        find = line.find(motif)
        return find != -1

    def _horizontal(self, action):
        column = self.board[:, action]
        motif = ('+' + str(-1 * self.player_turn)) * 4
        column = "".join(['+' + str(x) for x in column])
        # print('motif ', motif, 'column ', column)
        find = column.find(motif)
        return find != -1

    def _diag_principale(self, index, action):
        diag = list()
        i = index
        a = action
        while a > 0 and i < self.length - 1:
            a -= 1
            i += 1
        while a <= self.height - 1 and i >= 0:
            diag.append(str(self.board[i, a]))
            a += 1
            i -= 1
        motif = ('+' + str(-1 * self.player_turn)) * 4
        diag = "".join(['+' + str(x) for x in diag])
        find = diag.find(motif)
        return find != -1 and diag[find - 1] != '-'

    def _diag_reverse(self, index, action):
        diag = list()
        i = index
        a = action
        while a > 0 and i > 0:
            a -= 1
            i -= 1
        while a < self.height and i < self.length:
            diag.append(str(self.board[i, a]))
            a += 1
            i += 1
        motif = ('+' + str(-1 * self.player_turn)) * 4
        diag = "".join(['+' + str(x) for x in diag])
        find = diag.find(motif)
        return find != -1 and diag[find - 1] != '-'

    def __str__(self):
        return '\n'.join([''.join([self.corresp[y] for y in x]) for x in self.board.tolist()])

    def __repr__(self):
        return str(self.board)

    def render(self, logger):
        for row in self.board:
            logger.info(str([self.corresp[y] for y in row]))
        logger.info('-' * 50)
        # logger.info('\n{}'.format(self))
        # logger.info('-'*50)
