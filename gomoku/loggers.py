from utils import setup_logger
import os
import configparser

config = configparser.ConfigParser()
config.read('gomoku/config_gomoku.ini')
FOLDER = config['DEFAULT']['folder']

LOGGER_DISABLED = {'main': False,
                   'memory': False,
                   'tourney': False,
                   'model': False}

logger_main = setup_logger('logger_main_gomoku', os.path.join(FOLDER, 'logs/logger_main.log'))
logger_main.disabled = LOGGER_DISABLED['main']

logger_tourney = setup_logger('logger_tourney_gomoku', os.path.join(FOLDER, 'logs/logger_tourney.log'))
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger('logger_memory_gomoku', os.path.join(FOLDER, 'logs/logger_memory.log'))
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger('logger_model_gomoku', os.path.join(FOLDER, 'logs/logger_model.log'))
logger_model.disabled = LOGGER_DISABLED['model']
