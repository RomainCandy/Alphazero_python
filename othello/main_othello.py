import configparser

config = configparser.ConfigParser()
config.read('othello/config_othello.ini')

DEFAULT = config['DEFAULT']
FOLDER = DEFAULT['folder']
memory_file = DEFAULT['memory_file']
saved_model = DEFAULT['saved_model']
length = DEFAULT.getint('length')
height = DEFAULT.getint('height')
mcts_sim = DEFAULT.getint('mcts_sim')
num_self_play = DEFAULT.getint('num_self_play')
num_game_tournament = DEFAULT.getint('num_game_tournament')
turn_until_greedy = DEFAULT.getint('turn_until_greedy')
memory_size = DEFAULT.getint('memory_size')
threshold = DEFAULT.getfloat('threshold')
