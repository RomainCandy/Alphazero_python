import os

import connect4.loggers as lg
from Agent import Agent
from Memory import Memory
from connect4 import FOLDER, memory_file, saved_model, length, height, mcts_sim, num_self_play
from connect4 import num_game_tournament, turn_until_greedy, memory_size, threshold
from connect4.Connect4Game import Connect4Game as Game
from connect4.Connect4Game import StateConnect4 as State
from models import WrapperNet
from tournament import play_game
from utils import load_pickle, save_pickle, extract_digit

# from gomoku import FOLDER, memory_file, saved_model, length, height, mcts_sim, num_self_play
# from gomoku import num_game_tournament, turn_until_greedy, memory_size, threshold
# from gomoku.gomokuGame import GomokuGame as Game
# from gomoku.gomokuGame import StateGomoku as State


# from tictactoe.tictactoegame import StateTicTacToe as State
# from tictactoe.tictactoegame import GameTicTacToe as Game
# import tictactoe.loggers as lg
# from tictactoe.train_tictactoe import FOLDER, memory_file, saved_model, length, height, mcts_sim, num_self_play
# from tictactoe.train_tictactoe import num_game_tournament, turn_until_greedy, memory_size, threshold


# from connect4.main_connect4 import train


def train():
    try:
        memory = load_pickle(os.path.join(FOLDER, memory_file))
    except (EOFError, FileNotFoundError):
        memory = Memory(memory_size)
    env = Game(height, length)
    action_size = len(env.state.action_possible)
    model_contender = WrapperNet(env, lg.logger_model)
    model_champion = WrapperNet(env, lg.logger_model)
    try:
        model_champion.load_checkpoint(os.path.join(FOLDER, saved_model), 'best.pth.tar')
        last_version = max((x for x in os.listdir(os.path.join(FOLDER, saved_model))
                            if "version" in x), key=extract_digit)
        model_contender.load_checkpoint(os.path.join(FOLDER, saved_model), last_version)
        best_version = int(extract_digit(last_version))
    except FileNotFoundError:
        model_champion.save_checkpoint(folder=os.path.join(FOLDER, saved_model), filename='best.pth.tar')
        best_version = 1
    contender = Agent('CONTENDER', mcts_sim, action_size, 1, model_contender)
    champion = Agent('CHAMPION', mcts_sim, action_size, 1, model_champion)

    for it in range(50):
        lg.logger_main.info('=' * 10)
        lg.logger_main.info('ITERATION {}'.format(it))
        lg.logger_main.info('='*10)
        _, memory = play_game(champion, champion, env, num_self_play,
                              memory, turn_until_greedy, lg.logger_main)
        save_pickle(memory, os.path.join(FOLDER, memory_file))
        contender.train(memory, batch_size=256)
        model_contender.save_checkpoint(folder=os.path.join(FOLDER, saved_model),
                                        filename='version{}.pth.tar'.format(best_version))

        lg.logger_memory.info('====================')
        lg.logger_memory.info('NEW MEMORIES')
        lg.logger_memory.info('====================')

        memory_samp = memory.sample(50)

        for s in memory_samp:
            st, pi, v = s
            sta = State(st[0], st[1].mean().item())
            current_value, current_probs, _ = contender.get_preds(sta)
            best_value, best_probs, _ = champion.get_preds(sta)
            sta.render(lg.logger_memory)
            lg.logger_memory.info('MCTS VALUE : {:.2f}'.format(v))
            lg.logger_memory.info('CUR PRED VALUE FOR {}: {:.2f}'.format(sta.corresp[sta.player_turn], current_value))
            lg.logger_memory.info('BEST PRED VALUE FOR {}: {:.2f}'.format(sta.corresp[sta.player_turn], best_value))
            lg.logger_memory.info('THE MCTS ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in pi]))
            lg.logger_memory.info('CUR PRED ACTION VALUES: {}'.format(
                ['{:.2f}'.format(elem) for elem in current_probs]))
            lg.logger_memory.info('BES PRED ACTION VALUES: {}'.format(['{:.2f}'.format(elem) for elem in best_probs]))
            lg.logger_memory.info('*'*100)
        scores, _ = play_game(contender, champion, env, num_game_tournament, memory=None,
                              turns_until_greedy=0, logger=lg.logger_tourney)
        lg.logger_main.info('='*10)
        lg.logger_main.info('SCORE : {}'.format(scores))
        lg.logger_main.info('='*10)
        if scores['CONTENDER'] + scores['CHAMPION'] and (scores['CONTENDER'] / (scores['CONTENDER'] +
                                                                                scores['CHAMPION'])) > threshold:
            best_version += 1
            champion.model.net.load_state_dict(model_contender.net.state_dict())
            model_contender.save_checkpoint(folder=os.path.join(FOLDER, saved_model),
                                            filename='best.pth.tar')

            lg.logger_main.info('='*10)
            lg.logger_main.info('NEW BEST !')
            lg.logger_main.info("% of wins by the contender: {} ".format(scores['CONTENDER'] /
                                                                         (scores['CONTENDER'] + scores['CHAMPION'])))
            lg.logger_main.info('='*10)


if __name__ == '__main__':
    train()
