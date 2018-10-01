import numpy as np


def play_game(player1, player2, env, nb_games, memory, turns_until_greedy, logger):
    scores = {player1.name: 0, 'drawn': 0, player2.name: 0}
    sp = {'X': 0, 'O': 0}
    for idx_game in range(nb_games):
        logger.info('='*10)
        logger.info('game {} of {}'.format(idx_game + 1, nb_games))
        logger.info('='*10)
        done = 0
        turn = 0
        value = None
        player1.mcts = None
        player2.mcts = None
        if idx_game % 2:
            players = {1: {'agent': player1, 'name': player1.name},
                       -1: {'agent': player2, 'name': player2.name}}
        else:
            players = {-1: {'agent': player1, 'name': player1.name},
                       1: {'agent': player2, 'name': player2.name}}
        state = env.reset()
        mem = []
        if idx_game % 2:
            logger.info('{} play as X'.format(player1.name))
            logger.info('{} play as O'.format(player2.name))
        else:
            logger.info('{} play as X'.format(player2.name))
            logger.info('{} play as O'.format(player1.name))
        logger.info('-'*50)
        env.state.render(logger)
        while not done:
            turn += 1
            action, pi, mcts_value, nn_value = players[state.player_turn]['agent'].act(
                state, turn <= turns_until_greedy)
            logger.info(players[state.player_turn]['name'])
            logger.info('action: {}'.format(action))
            logger.info('pi : {}'.format(pi))
            logger.info('MCTS perceived value for {}: {}'.format(
                state.corresp[state.player_turn], np.round(mcts_value, 2)))
            logger.info('NN perceived value for {}: {}'.format(
                state.corresp[state.player_turn], np.round(nn_value, 2)))
            logger.info('='*10)
            if memory is not None:
                sym = state.get_symmetries(pi)
                for board, p in sym:
                    mem.append([board, p, None])
            state, value, done = env.step(action)
            # else:
            #     action, pi, mcts_value, nn_value = player2.act(state, turn <= turns_until_greedy)
            #     # print(state, action, pi, mcts_value, nn_value)
            #     # print(state)
            #     # print('*'*50)
            #     logger.info(player2.name)
            #     logger.info('action: {}'.format(action))
            #     logger.info('pi : {}'.format(pi))
            #     logger.info('MCTS perceived value for {}: {}'.format(
            #         state.corresp[state.player_turn], np.round(mcts_value, 2)))
            #     logger.info('NN perceived value for {}: {}'.format(
            #         state.corresp[state.player_turn], np.round(nn_value, 2)))
            #     logger.info('=' * 10)
            #     if memory is not None:
            #         mem.append([state.to_model(), pi, state.player_turn])
            #     state, value, done = env.step(action)
            env.state.render(logger)
        # logger.info('GAME : {}, VALUE: {}, NAME : {}'.format(idx_game % 2, value, env.player_turn))
        if value == 0:
            scores['drawn'] += 1
            logger.info('Draw!')
        elif value == 1:
            logger.info('WIN:  {}'.format(players[state.player_turn]['name']))
            scores[players[state.player_turn]['name']] += 1
            if state.player_turn == 1:
                sp['X'] += 1
            else:
                sp['O'] += 1
        elif value == -1:
            logger.info('WIN: {}'.format(players[-state.player_turn]['name']))
            scores[players[-state.player_turn]['name']] += 1
            if state.player_turn == 1:
                sp['O'] += 1
            else:
                sp['X'] += 1
        else:
            raise ValueError('value {} is not normal'.format(value))
        # else:
            # if idx_game % 2:
            #     if value == -1:
            #         scores[player2.name] += 1
            #         sp['X'] += 1
            #         logger.info('XX WINS!')
            #     else:
            #         scores[player1.name] += 1
            #         sp['O'] += 1
            #         logger.info('OO WINS!')
            # else:
            #     if value == -1:
            #         scores[player1.name] += 1
            #         sp['O'] += 1
            #         logger.info('O WINS!')
            #     else:
            #         sp['X'] += 1
            #         scores[player2.name] += 1
            #         logger.info('X WINS!')
        for elem in mem:
            elem[-1] = value * ((-1) ** (elem[-1] != state.player_turn))
        if memory is not None:
            memory.extend(mem)
    logger.info('SP {}'.format(sp))
    return scores, memory
