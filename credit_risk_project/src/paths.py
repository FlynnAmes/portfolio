""" computes path to root of project directory, as well as other paths referenced 
in the project"""

from pathlib import Path

BASE_PATH = Path(__file__).parent.parent
MODELS_PATH = BASE_PATH / 'models'
LOGS_PATH = BASE_PATH / 'logs'
CONFIG_PATH = BASE_PATH / 'config.yml'
DATA_PATH = BASE_PATH / 'data'
