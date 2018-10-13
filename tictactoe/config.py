import configparser


config = configparser.ConfigParser()
config['DEFAULT'] = {'length': 3,
                     'height': 3,
                     'mcts_sim': 20,
                     'num_self_play': 20,
                     'num_game_tournament': 10,
                     'turn_until_greedy': 2,
                     'threshold': .55,
                     'folder': 'tictactoe/',
                     'memory_file': 'memory/mem.p',
                     'saved_model': 'checkpoint',
                     'memory_size': 300}

with open('config_tictactoe.ini', 'w') as cf:
    config.write(cf)
