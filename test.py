from env import Game
from Agent import Agent
import numpy as np
from models import Net


class Model:
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, x):
        return np.ones(self.action_size) / self.action_size, 0


model = Net()

if __name__ == '__main__':
    # check for bug
    m = Model(7)
    env = Game(6, 7)
    agent = Agent('coucou', 1600, 7, 1, model)
    done = 0
    turn = 0
    state = env.reset()
    while not done:
        turn += 1
        if not turn % 2:
            action, pi, MCTS_value, NN_value = agent.act(state)
            # print(env.state.corresp[env.player_turn])
            # print('*' * 200)
            state, value, done = env.step(action)
            print(action, list((round(x, 2) for x in pi)), MCTS_value, NN_value)
            print(env.state)
            print('*'*200)
        else:
            while True:
                move = int(input('play a move'))
                try:
                    state, value, done = env.step(move)
                    break
                except ValueError:
                    print('forbidden move')
                    continue
            print(env.state)
    print(env.state.corresp[-1*env.player_turn], ' won')

