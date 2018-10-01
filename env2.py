#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:33:23 2017

@author: romain
"""

import numpy as np


class GameTicTacToe:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.board = np.zeros((length, height), dtype=int)
        self.player_turn = 1
        self.state = StateTicTacToe(self.board, self.player_turn)

    def reset(self):
        self.board = np.zeros((self.length, self.height), dtype=int)
        self.player_turn = 1
        self.state = StateTicTacToe(self.board, self.player_turn)
        return self.state

    def step(self, action):
        next_state, reward, done = self.state.take_action(action)
        self.state = next_state
        self.player_turn *= -1
        return next_state, reward, done

    def render(self):
        return str(self.state)


class StateTicTacToe:
    def __init__(self, board, player_turn):
        self.board = board
        self.length, self.height = board.shape
        self.action_possible = self._get_actions()
        # print(self.action_possible)
        self.player_turn = player_turn
        self.corresp = {1: 'X', -1: 'O', 0: '-'}
        self.id = self.__str__()

    def take_action(self, action):
        # action un entier
        if action not in self.action_possible:
            raise ValueError(self)
        # column = self.board[:, action]
        new_board = np.array(self.board)
        i = action % 3
        j = action // 3
        new_board[i, j] = self.player_turn
        next_state = StateTicTacToe(new_board, -1 * self.player_turn)
        value = 0
        done = 0
        if not len(next_state.action_possible):
            done = 1
        elif next_state.end_game():
            done = 1
            value = -1
        return next_state, value, done

    def _end_game(self):
        return self._column() or self._line() or self._diag()

    def _column(self):
        for i in range(self.length):
            if self.board[i, 0] and all([x == self.board[i, 0] for x in self.board[i]]):
                return True
        return False

    def end_game(self):
        if self.board[0, 0] and (self.board[0, 0] == self.board[0, 1] == self.board[0, 2]):
            return True
        elif self.board[0, 1] and (self.board[0, 1] == self.board[1, 1] == self.board[2, 1]):
            return True
        elif self.board[0, 2] and (self.board[0, 2] == self.board[1, 2] == self.board[2, 2]):
            return True
        elif self.board[0, 0] and (self.board[0, 0] == self.board[1, 0] == self.board[2, 0]):
            return True
        elif self.board[1, 0] and (self.board[1, 0] == self.board[1, 1] == self.board[1, 2]):
            return True
        elif self.board[2, 0] and (self.board[2, 0] == self.board[2, 1] == self.board[2, 2]):
            return True
        elif self.board[0, 0] and (self.board[0, 0] == self.board[1, 1] == self.board[2, 2]):
            return True
        elif self.board[2, 0] and (self.board[2, 0] == self.board[1, 1] == self.board[0, 2]):
            return True
        return False

    def _diag(self):
        return  self.board[1, 1] and ((self.board[0, 0] == self.board[1, 1] == self.board[2, 2]) or (
                self.board[2, 0] == self.board[1, 1] == self.board[0, 2]))

    def to_model(self):
        board = np.zeros((2, self.length, self.height))
        board[0] = self.board
        board[1] = self.player_turn
        return board

    def _get_actions(self):
        zz = np.argwhere(self.board == 0)
        return [x + 3 * y for x, y in zz]

    def __str__(self):
        return '\n'.join([''.join([self.corresp[y] for y in x]) for x in self.board.tolist()])

    def __repr__(self):
        return str(self.board)

import random as rd
def play():
    # import time
    done = 0
    turn = 0
    env = GameTicTacToe(3, 3)
    state = env.state
    reward = None
    # print(env.render())
    # print('-' * 50)
    while done == 0:
        print(env.render())
        print('-' * 50)
        turn += 1
        action = rd.choice(state.action_possible)
        # time.sleep(1)
        state, reward, done = env.step(action)
        # print(reward)
    # print(reward)
    print(env.render())
    print('-'*100)
    # print(env.player_turn)
    if not reward:
        print('draw')
    else:
        print('winner is ', env.state.corresp[-1*env.player_turn])


if __name__ == '__main__':
    play()
