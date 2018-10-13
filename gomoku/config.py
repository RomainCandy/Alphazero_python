import configparser


config = configparser.ConfigParser()
config['DEFAULT'] = {'length': 8,
                     'height': 8,
                     'mcts_sim': 400,
                     'num_self_play': 100,
                     'num_game_tournament': 100,
                     'turn_until_greedy': 20,
                     'threshold': .55,
                     'folder': 'gomoku/',
                     'memory_file': 'memory/mem.p',
                     'saved_model': 'checkpoint',
                     'memory_size': 100000}

with open('config_gomoku.ini', 'w') as cf:
    config.write(cf)
