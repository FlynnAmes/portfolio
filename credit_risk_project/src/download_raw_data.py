""" Downloads data automtically from Kaggle. 
Note this requires Kaggle API jeys to be set up """

import os
from kaggle.api.kaggle_api_extended import KaggleApi
from src.paths import DATA_PATH


def load_data_from_kaggle():
    """ load data to raw data directory from kaggle"""
    if not os.path.exists(DATA_PATH / 'raw' / 'credit_risk_benchmark_dataset.csv'):
        # instance of kaggle API
        api = KaggleApi()
        # authenticate API keys (note that will need to set up in environment)
        api.authenticate()
        # download the csv file and save to raw data directory
        api.dataset_download_files("adilshamim8/credit-risk-benchmark-dataset", path=DATA_PATH / 'raw', unzip=True)
        # rename file to put underscores in place of spaces
        os.rename(DATA_PATH / 'raw' / 'Credit Risk Benchmark Dataset.csv', DATA_PATH / 'raw' / 'credit_risk_benchmark_dataset.csv')

if __name__ == '__main__':
    load_data_from_kaggle()
