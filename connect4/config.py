import configparser


config = configparser.ConfigParser()
config['DEFAULT'] = {'length': 7,
                     'height': 6,
                     'mcts_sim': 150,
                     'num_self_play': 200,
                     'num_game_tournament': 50,
                     'turn_until_greedy': 10,
                     'threshold': .55,
                     'folder': 'connect4/',
                     'memory_file': 'memory/mem.p',
                     'saved_model': 'checkpoint',
                     'memory_size': 15000}

with open('config_connect4.ini', 'w') as cf:
    config.write(cf)
