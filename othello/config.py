import configparser


config = configparser.ConfigParser()
config['DEFAULT'] = {'length': 8,
                     'height': 8,
                     'mcts_sim': 200,
                     'num_self_play': 100,
                     'num_game_tournament': 100,
                     'turn_until_greedy': 15,
                     'threshold': .55,
                     'folder': 'othello/',
                     'memory_file': 'memory/mem.p',
                     'saved_model': 'checkpoint',
                     'memory_size': 300000}

with open('config_othello.ini', 'w') as cf:
    config.write(cf)
