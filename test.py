from env import Game
from Agent import Agent
from tournament import play_game
import numpy as np
from models import WrapperNet
from Memory import Memory
from datetime import datetime
import loggers
from env2 import GameTicTacToe


class Model:
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, x):
        return np.ones(self.action_size) / self.action_size, 0*x


if __name__ == '__main__':
    # check for bug
    # m = Model(9)
    # env = GameTicTacToe(3, 3)
    # agent = Agent('coucou', 10, 9, 1, m)
    # done = 0
    # turn = 0
    # state = env.reset()
    # while not done:
    #     turn += 2
    #     if not turn % 2:
    #         action, pi, MCTS_value, NN_value = agent.act(state, 0)
            # print(env.state.corresp[env.player_turn])
            # print('*' * 200)
            # state, value, done = env.step(action)
            # print(action, list((round(x, 2) for x in pi)), MCTS_value, NN_value)
            # print(env.state)
            # print('*'*200)
        # else:
        #     while True:
        #         move = int(input('play a move'))
        #         try:
        #             state, value, done = env.step(move)
        #             break
        #         except ValueError:
        #             print('forbidden move')
        #             continue
        #     print(env.state)
    # end = int(MCTS_value)
    # print(value)
    # if end == 0:
    #     print('draw')
    # else:
    #     print(env.state.corresp[-1*env.player_turn], ' won')
    memory = Memory(100000)
    length = 7
    height = 6
    tournament = 400
    turn_until_greedy = 20
    env = Game(height, length)
    model_contender = WrapperNet(env)
    model_champion = WrapperNet(env)
    model_champion.save_checkpoint(folder='checkpoint', filename='best.pth.tar')
    # model_contender.load_checkpoint(folder='checkpoint', filename='version4.pth.tar')
    # model_champion.load_checkpoint(folder='checkpoint', filename='version10.pth.tar')
    contender = Agent('CONTENDER', 100, length, 1, model_contender)
    champion = Agent('CHAMPION', 100, length, 1, model_champion)
    best_version = 1
    print('start at {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    for it in range(1, 100):
        _, memory = play_game(champion, champion, env, 100, memory, turn_until_greedy)
        print('self.play {} finished at {}'.format(it, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        model_contender.train(memory, 64)
        model_contender.save_checkpoint(folder='checkpoint', filename='version{}.pth.tar'.format(best_version))
        print('training {} finished at {}'.format(it, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('tournament {} beginned at {}'.format(it, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        scores, _ = play_game(contender, champion, env, tournament, memory=None,
                              turns_until_greedy=0)
        print('tournament {} finished at {}'.format(it, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print(scores)
        # if tournament - scores['drawn'] and scores['CONTENDER'] / (tournament - scores['drawn']) > 0.55:
        if scores['CONTENDER'] * .55 > scores['CHAMPION']:
            print('*'*100)
            print('*'*100)
            print('\t\t\tBEST !')
            print('*'*100)
            print('*'*100)
            best_version += 1
            champion.model.net.load_state_dict(model_contender.net.state_dict())
            model_contender.save_checkpoint(folder='checkpoint', filename='best.pth.tar')
    # m = Model(9)
    # env = GameTicTacToe(3, 3)
    # agent = Agent('coucou', 10, 9, 1, m)
    # print(play_game(champion, agent, env, 1, None, 0))
