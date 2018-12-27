import numpy as np
import sys
sys.path.append('..')
from GenericGame import GenericState


class OthelloGame:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.action_size = length * height
        self.board = np.zeros((length, height), dtype=int)
        self.player_turn = 1
        self.board[length // 2, height // 2] = self.board[length // 2 - 1, height // 2 - 1] = 1
        self.board[length // 2 - 1, height // 2] = self.board[length // 2, height // 2 - 1] = -1
        self.state = StateOthello(self.board, self.player_turn)

    def reset(self):
        self.board = np.zeros((self.length, self.height), dtype=int)
        self.player_turn = 1
        self.board[self.length // 2, self.height // 2] = self.board[self.length // 2 - 1, self.height // 2 - 1] = 1
        self.board[self.length // 2 - 1, self.height // 2] = self.board[self.length // 2, self.height // 2 - 1] = -1
        self.state = StateOthello(self.board, self.player_turn)
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


class StateOthello(GenericState):

    def __init__(self, board, player_turn):
        self.player_turn = player_turn
        super(StateOthello, self).__init__(board, player_turn)
        self.over = not (len(self.action_possible))

    def is_draw(self):
        if self.is_over():
            return self.board.sum() == 0
        return False

    def is_lost(self, action=None):
        if self.is_over():
            return (self.board.sum() * self.player_turn) < 0
        return False

    def is_win(self, action=None):
        if self.is_over():
            return (self.board.sum() * self.player_turn) > 0
        return False

    def next_state(self):
        # for action in self.action_possible:
        for action in self.action_possible:
            yield action, self.take_action(action)[0]

    def is_terminal(self, action):
        return self.is_over()

    def evaluate(self):
        return self.board.sum() * self.player_turn

    def take_action(self, action):
        # assert action in self.action_possible
        new_board = np.array(self.board)
        j = action % self.length
        i = action // self.height
        new_board[i, j] = self.player_turn
        idx = self._horizontal(i, j) + self._vertical(i, j) + self._diag_principal(i, j) + self._diag_reverse(i, j)
        for x in idx:
            new_board[x] = self.player_turn
        next_state = StateOthello(new_board, -1 * self.player_turn)
        if next_state.over:
            next_state = StateOthello(new_board, self.player_turn)
            if next_state.over:
                next_state = StateOthello(new_board, -1 * self.player_turn)
        value = 0
        done = 0
        if next_state.is_over():
            done = 1
            res = np.sum(next_state.board) * self.player_turn * -1
            if res == 0:
                value = 0
            else:
                value = 2 * int(res > 0) - 1
        return next_state, value, done

    def is_over(self):
        return self.over or 0 not in self.board

    def get_symmetries(self, pi):
        return [(self.to_model(), pi)]

    def _get_actions(self):
        act = np.argwhere(self.board == 0)
        zz = [(x, y) for x, y in act if self._is_valid(x, y)]
        return np.array([y + self.length * x for x, y in zz])

    def _is_valid(self, x, y):
        # if self._vertical(x, y):
        #     print("vertical ", x, y)
        #     print("flip ", self._vertical(x, y))
        #     return True
        # if self._horizontal(x, y):
        #     print("horizontal ", x, y)
        #     print("flip ", self._horizontal(x, y))
        #     return True
        # if self._diag_principal(x, y):
        #     print("principal ", x, y)
        #     print("flip ", self._diag_principal(x, y))
        #     return True
        # if self._diag_reverse(x, y):
        #     print("reverse ", x, y)
        #     print("flip ", self._diag_reverse(x, y))
        #     return True
        # return False
        return (self._vertical(x, y) or self._horizontal(x, y)
                or self._diag_principal(x, y) or self._diag_reverse(x, y))

    def _horizontal(self, index, y):
        n = self.board.shape[0]
        temp_right = []
        temp_left = []
        if y < n - 1:
            line_right = self.board[index, y:]
            if line_right[1] == -1 * self.player_turn:
                for i, elem in enumerate(line_right[1:], 1):
                    if elem == self.player_turn:
                        break
                    temp_right.append((index, y + i))
                else:
                    temp_right.clear()
        if y > 1:
            line_left = self.board[index, :y]
            if line_left[-1] == -1 * self.player_turn:
                for i, elem in enumerate(reversed(line_left), 1):
                    if elem == self.player_turn:
                        break
                    temp_left.append((index, y - i))
                else:
                    temp_left.clear()
        return temp_left + temp_right

    def _vertical(self, x, index):
        temp_right = []
        temp_left = []
        n = self.board.shape[0]
        if x < n - 1:
            line_right = self.board[x:, index]
            if line_right[1] == -1 * self.player_turn:
                for i, elem in enumerate(line_right[1:], 1):
                    if elem == self.player_turn:
                        break
                    temp_right.append((x + i, index))
                else:
                    temp_right.clear()
        if x > 1:
            line_left = self.board[:x, index]
            if line_left[-1] == -1 * self.player_turn:
                for i, elem in enumerate(reversed(line_left), 1):
                    if elem == self.player_turn:
                        break
                    temp_left.append((x - i, index))
                else:
                    temp_left.clear()
        # if temp_right + temp_left:
        #     print(temp_right, line_right, temp_left, line_left, index)
        return temp_left + temp_right

    def _diag_principal(self, i, j):
        n = self.board.shape[0]
        temp_up = []
        temp_d = []
        # diag = np.diag(self.board[:, ::-1], n - j - 1 - i)
        if i > 1 and j < n - 1:
            diag_up = [self.board[ix, jx] for ix, jx in zip(range(i, -1, -1), range(j, n))]
            if diag_up[1] == -1 * self.player_turn:
                for idx, elem in enumerate(diag_up[1:], 1):
                    # print(idx, elem, diag_up)
                    if elem == self.player_turn:
                        break
                    temp_up.append((i - idx, j + idx))
                else:
                    temp_up.clear()

        if i < n - 1 and j > 1:
            diag_d = [self.board[ix, jx] for ix, jx in zip(range(i, n), range(j, -1, -1))]
            if diag_d[1] == -1 * self.player_turn:
                for idx, elem in enumerate(diag_d, 1):
                    if elem == self.player_turn:
                        break
                    temp_d.append((i + idx, j - idx))
                else:
                    temp_d.clear()
        return temp_d + temp_up

    def _diag_reverse(self, i, j):
        n = self.board.shape[0]
        temp_up = []
        temp_d = []
        if i < n - 1 and j < n - 1:
            diag_up = [self.board[ix, jx] for ix, jx in zip(range(i, n), range(j, n))]
            if diag_up[1] == -1 * self.player_turn:
                for idx, elem in enumerate(diag_up, 1):
                    if elem == self.player_turn:
                        break
                    temp_up.append((i + idx, j + idx))
                else:
                    temp_up.clear()
        if i > 1 and j > 1:
            diag_d = [self.board[ix, jx] for ix, jx in zip(range(i, -1,  -1), range(j, -1, -1))]
            if diag_d[1] == -1 * self.player_turn:
                for idx, elem in enumerate(diag_d, 1):
                    if elem == self.player_turn:
                        break
                    temp_d.append((i - idx, j - idx))
                else:
                    temp_d.clear()
        return temp_up + temp_d

    def _get_corresp(self):
        return {1: 'X', -1: 'O', 0: '-'}

    def __str__(self):
        return '\n'.join([''.join([self.corresp[y] for y in x]) for x in self.board.tolist()])\
               + str(self.player_turn)

    def __repr__(self):
        return str(self.board)


if __name__ == "__main__":
    def main():
        import random as rd
        # zz = 'X----X--XOOOOOOOXOXXXOOXXXOXOXOXXOXXXXXXXOOOOXXXX-O-OOXXX---OXXX'
        # board = list()
        # for x in range(8):
        #     temp = list()
        #     for y in range(8):
        #         temp.append({'X': 1, 'O': -1, '-': 0}[zz[x * 8 + y]])
        #     board.append(temp)
        # board = np.array(board)
        # state = StateOthello(board, -1)
        # print(state)
        # print(state.over)
        # print(state.action_possible)
        N = 8
        board = np.zeros(shape=(N, N))
        board[N // 2, N // 2] = board[N // 2 - 1, N // 2 - 1] = 1
        board[N // 2 - 1, N // 2] = board[N // 2, N // 2 - 1] = -1
        state = StateOthello(board, 1)
        print(state)
        d = False
        for _ in range(N*N):
            if d:
                break
            print("*" * 50)
            act = state.action_possible
            state, r, d = state.take_action(rd.choice(act))
            print(state)
        if r == 1:
            print(state.corresp[state.player_turn], " won")
        elif r == -1:
            print(state.corresp[state.player_turn * -1], " won")
        else:
            print("Draw!")
    main()
