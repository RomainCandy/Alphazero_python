import numpy as np


def play_game(player1, player2, env, nb_games, memory, turns_until_greedy, logger):
    """
    Args:
        player1: either an Agent or a User to play.
        player2: either an Agent or an User to play.
        env: A game.
        nb_games: nb_games to play.
        memory: A memory object or None.
        turns_until_greedy: turns where the action become greedy
                            with respect to the policy.
        logger: a logger object to log games and results.

    Returns: tuple
        scores as a dictionary with the number of wins
        for player1, player 2 and the number of draws

        the memory of the played games if memory is not None else None
    """
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
            logger.info('pi : {}'.format([round(x, 3) for x in pi]))
            logger.info('MCTS perceived value for {}: {:.4f}'.format(
                state.corresp[state.player_turn], np.round(mcts_value, 2)))
            logger.info('NN perceived value for {}: {:.4f}'.format(
                state.corresp[state.player_turn], np.round(nn_value, 2)))
            logger.info('='*10)
            if memory is not None:
                sym = state.get_symmetries(pi)
                for board, p in sym:
                    mem.append([board, p, state.player_turn])
            state, value, done = env.step(action)
            env.state.render(logger)
        # logger.info('GAME : {}, VALUE: {}, NAME : {}'.format(idx_game % 2, value, env.player_turn))
        if value == 0:
            scores['drawn'] += 1
            logger.info('Draw!')
        elif value == 1:
            logger.info('WIN:  {} in {} turns'.format(players[state.player_turn]['name'], turn))
            scores[players[state.player_turn]['name']] += 1
            if state.player_turn == 1:
                sp['X'] += 1
            else:
                sp['O'] += 1
        elif value == -1:
            logger.info('WIN: {} in {} turns'.format(players[-state.player_turn]['name'], turn))
            scores[players[-state.player_turn]['name']] += 1
            if state.player_turn == 1:
                sp['O'] += 1
            else:
                sp['X'] += 1
        else:
            raise ValueError('value {} is not normal'.format(value))
        for elem in mem:
            elem[-1] = value * ((-1) ** (elem[-1] != state.player_turn))
        if memory is not None:
            memory.extend(mem)
    logger.info('SP {}'.format(sp))
    return scores, memory


def friendly_game(player1, player2, env):
    turn = 0
    done = 0
    value = None
    player1.mcts = None
    player2.mcts = None
    players = {1: {'agent': player1, 'name': player1.name},
               -1: {'agent': player2, 'name': player2.name}}
    state = env.reset()
    print(env)
    print('*'*100)
    while not done:
        turn += 1
        action, _, _, _ = players[state.player_turn]['agent'].act(state, 0)
        state, value, done = env.step(action)
        print(env)
        print("*"*100)
    if value == 0:
        print('Draw!')
    elif value == 1:
        print('{} WON !'.format(players[state.player_turn]['name']))
    elif value == -1:
        print('{} WON !'.format(players[-state.player_turn]['name']))
    else:
        raise ValueError('value {} is not normal'.format(value))
