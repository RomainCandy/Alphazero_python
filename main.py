import loggers
from Agent import Agent
from tournament import play_game
from env import Game, State
import numpy as np
from models import WrapperNet
from Memory import Memory
import loggers as lg
import pickle


class Model:
    def __init__(self, action_size):
        self.action_size = action_size

    def predict(self, x):
        return np.ones(self.action_size) / self.action_size, 0


def main():
    memory = Memory(100000)
    length = 7
    height = 6
    mcts_sim = 160
    num_self_play = 250
    num_game_tournament = 40
    turn_until_greedy = 20
    threshold = .55
    env = Game(height, length)
    action_size = len(env.state.action_possible)
    model_contender = WrapperNet(env)
    model_champion = WrapperNet(env)
    model_champion.save_checkpoint(folder='checkpoint', filename='best.pth.tar')
    contender = Agent('CONTENDER', mcts_sim, action_size, 1, model_contender)
    champion = Agent('CHAMPION', mcts_sim, action_size, 1, model_champion)
    best_version = 1
    ok = False
    for it in range(50):
        lg.logger_main.info('=' * 10)
        lg.logger_main.info('ITERATION {}'.format(it))
        lg.logger_main.info('='*10)
        _, memory = play_game(champion, champion, env, num_self_play,
                              memory, turn_until_greedy, loggers.logger_main)
        memory.shuffle()
        with open('memory/mem.p', 'wb') as pfile:
            pickle.dump(memory, pfile)
        model_contender.train(memory, 64)
        model_contender.save_checkpoint(folder='checkpoint', filename='version{}.pth.tar'.format(best_version))

        lg.logger_memory.info('====================')
        lg.logger_memory.info('NEW MEMORIES')
        lg.logger_memory.info('====================')

        memory_samp = memory.sample(50)

        for s in memory_samp:
            st, pi, v = s
            sta = State(st[0], st[1].mean().item())
            current_value, current_probs, _ = contender.get_preds(sta)
            best_value, best_probs, _ = champion.get_preds(sta)
            lg.logger_memory.info('MCTS VALUE FOR \n {}: {:.2f}'.format(st[0], v))
            lg.logger_memory.info('CUR PRED VALUE FOR {}: {:.2f}'.format(sta.corresp[sta.player_turn], current_value))
            lg.logger_memory.info('BEST PRED VALUE FOR {}: {:.2f}'.format(sta.corresp[sta.player_turn], best_value))
            lg.logger_memory.info('THE MCTS ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in pi]))
            lg.logger_memory.info('CUR PRED ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in current_probs]))
            lg.logger_memory.info('BES PRED ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in best_probs]))
            sta.render(lg.logger_memory)
            lg.logger_memory.info('*'*100)
        scores, _ = play_game(contender, champion, env, num_game_tournament, memory=None,
                              turns_until_greedy=0, logger=lg.logger_tourney)
        lg.logger_main.info('='*10)
        lg.logger_main.info('SCORE : {}'.format(scores))
        lg.logger_main.info('='*10)
        if scores['CONTENDER'] + scores['CHAMPION'] and scores['CONTENDER'] / (scores['CONTENDER'] +
                                                                               scores['CHAMPION']) > threshold:
            best_version += 1
            champion.model.net.load_state_dict(model_contender.net.state_dict())
            model_contender.save_checkpoint(folder='checkpoint', filename='best.pth.tar')
            lg.logger_main.info('='*10)
            lg.logger_main.info('NEW BEST !')
            lg.logger_main.info('='*10)


if __name__ == '__main__':
    main()
