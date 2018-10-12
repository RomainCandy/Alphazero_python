import configparser


config = configparser.ConfigParser()
config['DEFAULT'] = {'length': 3,
                     'height': 3,
                     'mcts_sim': 50,
                     'num_self_play': 40,
                     'num_game_tournament': 20,
                     'turn_until_greedy': 2,
                     'threshold': .55,
                     'folder': 'tictactoe/',
                     'memory_file': 'memory/mem.p',
                     'saved_model': 'checkpoint',
                     'memory_size': 100}

with open('config_tictactoe.ini', 'w') as cf:
    config.write(cf)
