from utils import setup_logger
import os
import configparser

config = configparser.ConfigParser()
config.read('othello/config_othello.ini')
FOLDER = config['DEFAULT']['folder']
HEIGHT = config['DEFAULT']['length']

LOGGER_DISABLED = {'main': False,
                   'memory': False,
                   'tourney': False,
                   'model': False}

logger_main = setup_logger(f'logger_main_othello{HEIGHT}',
                           os.path.join(FOLDER, f'logs/logger_main{HEIGHT}.log'))
logger_main.disabled = LOGGER_DISABLED['main']

logger_tourney = setup_logger(f'logger_tourney_othello{HEIGHT}',
                              os.path.join(FOLDER, f'logs/logger_tourney{HEIGHT}.log'))
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger(f'logger_memory_othello{HEIGHT}',
                             os.path.join(FOLDER, f'logs/logger_memory{HEIGHT}.log'))
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger(f'logger_model_othello{HEIGHT}',
                            os.path.join(FOLDER, f'logs/logger_model{HEIGHT}.log'))
logger_model.disabled = LOGGER_DISABLED['model']
