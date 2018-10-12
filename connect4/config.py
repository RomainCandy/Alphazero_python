import configparser


config = configparser.ConfigParser()
config['DEFAULT'] = {'length': 7,
                     'height': 6,
                     'mcts_sim': 100,
                     'num_self_play': 100,
                     'num_game_tournament': 100,
                     'turn_until_greedy': 10,
                     'threshold': .55,
                     'folder': 'connect4/',
                     'memory_file': 'memory/mem.p',
                     'saved_model': 'checkpoint',
                     'size_memory': 30000}

with open('config_connect4.ini', 'w') as cf:
    config.write(cf)
