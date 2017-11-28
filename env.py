#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 08:33:23 2017

@author: romain
"""

import numpy as np
import random as rd
import time


class State:
    def __init__(self,height=6,width=7):
        self.height = height
        self.width = width
        self.board = np.zeros((self.height,self.width),dtype=int)
        self.action_possible = list(range(self.width))
        self.player = 1
        self.finished = False
        self.winner = None
        self.reward = 0
        self.corresp = {1:'X',-1:'O',0:'-'}
        
    def get_actions(self):
        return self.action_possible
    
    def step(self,action):
#        action un entier
        assert action in self.action_possible
        column = self.board[:,action]
        for index,pawn in reversed(list(enumerate(column))):
            if not pawn:
                self.board[index,action] = self.player
                if index == 0:
                    self.action_possible.remove(action)
                break
        if self.connect4(index,action):
            self.winner = self.corresp[self.player]
            self.reward =  self.winner
            self.finished = True
            
        if not self.action_possible:
            self.reward = 0
            self.finished = True
        self.player *=-1
    
    
    def vertical(self,index,action):
        line = self.board[index,:]
        motif = ('+'+str(self.player))*4
        line = "".join(['+'+str(x) for x in line])
        find = line.find(motif)
        return find != -1
    
    def horizontal(self,index,action):
        column = self.board[:,action]
        motif = ('+'+str(self.player))*4
        column = "".join(['+'+str(x) for x in column])
        find = column.find(motif)
        return find != -1
    
    def diag_principale(self,index,action):
        diag = list()
        i = index
        a = action
        while a > 0 and i < self.height-1:
            a -= 1
            i += 1
        while a<=self.width-1 and i >= 0:
            diag.append(str(self.board[i,a]))
            a += 1
            i -= 1
        motif = ('+'+str(self.player))*4
        diag = "".join(['+'+str(x) for x in diag])
        find = diag.find(motif)
        return find != -1 and diag[find-1] != '-'
    
    def diag_reverse(self,index,action):
        diag = list()
        i = index
        a = action
        while a > 0 and i > 0:
            a -= 1
            i -= 1
        while a<self.width and i < self.height:
            diag.append(str(self.board[i,a]))
            a += 1
            i += 1
        motif = ('+'+str(self.player))*4
        diag = "".join(['+'+str(x) for x in diag])
        find = diag.find(motif)
        return find != -1 and diag[find-1] != '-'
    
    def connect4(self,index,action):
        if self.horizontal(index,action):
            return True
        elif self.vertical(index,action):
            return True
        elif self.diag_principale(index,action):
            return True
        elif self.diag_reverse(index,action):
            return True
        return False
                
    def __str__(self):
        return '\n'.join([''.join([self.corresp[y] for y in x ]) for x in self.board.tolist()])
    
    def __repr__(self):
        return str(self.board)
    

def play():
#    random policy to check for bugs
    jeu = State()
    while not jeu.finished:
        action = rd.choice(jeu.action_possible)
        jeu.step(action)
        print(jeu)
        print("*"*50)
        time.sleep(.05)
    return jeu.winner,jeu,action

x,y,z=play()
print(y)
print(x,'won')
print('last action' , z)