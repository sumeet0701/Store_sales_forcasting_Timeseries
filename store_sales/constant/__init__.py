import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR = os.getcwd()  #to get current working directory
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

TIME_CONFIG_FILE_NAME='time_config.yaml'
TIME_CONFIG_FILE_PATH=os.path.join(ROOT_DIR,CONFIG_DIR,TIME_CONFIG_FILE_NAME)

EXOG_COLUMNS='exog_columns'

TARGET_COLUMN='target_column'
ENCODE_COLUMNS='encoding'



from store_sales.constant.training_pipeline import *
from store_sales.constant.data_base import *