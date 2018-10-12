# import loggers
# from Agent import Agent
# from tournament import play_game
# from env import Game, State
# from models import WrapperNet
# from Memory import Memory
# import loggers as lg
# import pickle
# import os
# from utils import load_pickle, save_pickle
# import re

from connect4.main_connect4 import train

# class Model:
#     def __init__(self, action_size):
#         self.action_size = action_size
#
#     def predict(self, x):
#         return np.ones(self.action_size) / self.action_size, 0


# FOLDER = '.'
# saved_model = 'checkpointV2'
# memory_file = 'memory/V2mem.p'
#
#
# def extract_digit(s):
#     return re.findall('\d+', s)[-1]
#
#
# def main():
#     try:
#         # with open('memory/mem.p', 'rb') as pfile:
#         #     memory = pickle.load(pfile)
#         memory = load_pickle(os.path.join(FOLDER, memory_file))
#     except (EOFError, FileNotFoundError):
#         memory = Memory(100000)
#     length = 7
#     height = 6
#     mcts_sim = 100
#     num_self_play = 100
#     num_game_tournament = 100
#     turn_until_greedy = 10
#     threshold = .55
#     env = Game(height, length)
#     action_size = len(env.state.action_possible)
#     model_contender = WrapperNet(env)
#     model_champion = WrapperNet(env)
#     try:
#         model_champion.load_checkpoint(os.path.join(FOLDER, saved_model), 'best.pth.tar')
#         last_version = max((x for x in os.listdir(os.path.join(FOLDER, saved_model))
#                             if "version" in x), key=extract_digit)
#         model_contender.load_checkpoint(os.path.join(FOLDER, saved_model), last_version)
#         best_version = int(extract_digit(last_version))
#     except FileNotFoundError:
#         model_champion.save_checkpoint(folder=os.path.join(FOLDER, saved_model), filename='best.pth.tar')
#         best_version = 1
#     contender = Agent('CONTENDER', mcts_sim, action_size, 1, model_contender)
#     champion = Agent('CHAMPION', mcts_sim, action_size, 1, model_champion)
#
#     for it in range(50):
#         lg.logger_main.info('=' * 10)
#         lg.logger_main.info('ITERATION {}'.format(it))
#         lg.logger_main.info('='*10)
#         _, memory = play_game(champion, champion, env, num_self_play,
#                               memory, turn_until_greedy, loggers.logger_main)
#         # memory.shuffle()
#         # with open(os.path.join(FOLDER, 'memory/V2mem.p'), 'wb') as pfile:
#         #     pickle.dump(memory, pfile)
#         save_pickle(memory, os.path.join(FOLDER, memory_file))
#         # model_contender.train(memory, 64)
#         contender.train(memory, batch_size=256)
#         model_contender.save_checkpoint(folder=os.path.join(FOLDER, saved_model),
#                                         filename='V2version{}.pth.tar'.format(best_version))
#
#         lg.logger_memory.info('====================')
#         lg.logger_memory.info('NEW MEMORIES')
#         lg.logger_memory.info('====================')
#
#         memory_samp = memory.sample(50)
#
#         for s in memory_samp:
#             st, pi, v = s
#             sta = State(st[0], st[1].mean().item())
#             current_value, current_probs, _ = contender.get_preds(sta)
#             best_value, best_probs, _ = champion.get_preds(sta)
#             sta.render(lg.logger_memory)
#             lg.logger_memory.info('MCTS VALUE : {:.2f}'.format(v))
#             lg.logger_memory.info('CUR PRED VALUE FOR {}: {:.2f}'.format(sta.corresp[sta.player_turn], current_value))
#             lg.logger_memory.info('BEST PRED VALUE FOR {}: {:.2f}'.format(sta.corresp[sta.player_turn], best_value))
#             lg.logger_memory.info('THE MCTS ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in pi]))
#             lg.logger_memory.info('CUR PRED ACTION VALUES: {}'.format(
#                 ['{:.2f}'.format(elem) for elem in current_probs]))
#             lg.logger_memory.info('BES PRED ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in best_probs]))
#             lg.logger_memory.info('*'*100)
#         scores, _ = play_game(contender, champion, env, num_game_tournament, memory=None,
#                               turns_until_greedy=0, logger=lg.logger_tourney)
#         lg.logger_main.info('='*10)
#         lg.logger_main.info('SCORE : {}'.format(scores))
#         lg.logger_main.info('='*10)
#         if scores['CONTENDER'] + scores['CHAMPION'] and (scores['CONTENDER'] / (scores['CONTENDER'] +
#                                                                                 scores['CHAMPION'])) > threshold:
#             best_version += 1
#             champion.model.net.load_state_dict(model_contender.net.state_dict())
#             model_contender.save_checkpoint(folder=os.path.join(FOLDER, saved_model),
#                                             filename='best.pth.tar')
#
#             lg.logger_main.info('='*10)
#             lg.logger_main.info('NEW BEST !')
#             lg.logger_main.info("% of wins by the contender: {} ".format(scores['CONTENDER'] /
#                                                                          (scores['CONTENDER'] + scores['CHAMPION'])))
#             lg.logger_main.info('='*10)
#

if __name__ == '__main__':
    train()
