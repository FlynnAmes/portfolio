""" Ingest and filter out bad data, save seperate validation and training data """

import os
import yaml
import json
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import pickle as pkl
from src.paths import CONFIG_PATH, DATA_PATH, LOGS_PATH


#TODO: rename validate data to test data (to match neural net naming convention)

def ingest_and_clean_data(config_path):

    # get random seed and training proportion for train test split
    with open(config_path) as f:
        config = yaml.safe_load(f)

    random_seed = config['random_seed']
    train_size = config['train_size']
    train_prop_no_tune = config['train_prop_no_tune']


    # load in data
    df = pd.read_csv(DATA_PATH / 'raw' / 'credit_risk_benchmark_dataset.csv')

    #############
    # Filtering
    #############

    print('\n', 'initial shape:', df.shape, '\n')

    # first removing revolving utilisation less than 5 - threshold determined from EDA
    df_filtered = df[df['rev_util'] < 5]

    # Next filter out bad values for num times late payment (again determined from EDA)
    df_filtered = df_filtered[(df['late_90'] != 98) & (df['late_90'] != 96)]

    # filter out bad values for monthly income (where equal to 1 or zero - very likely an error)
    mask_inc_bad_data = (df_filtered['monthly_inc'] == 1) | ((df_filtered['monthly_inc'] == 0) 
                                                            & (df_filtered['debt_ratio'] != 0))
    df_filtered = df_filtered[~mask_inc_bad_data]

    # create mask where extreme debt ratio coincide with unlikely monthly income
    mask_debt_ratio_bad_data = ((zscore(df_filtered['debt_ratio']) > 5) & (df_filtered['monthly_inc'] < 500))
    # filter data using mask
    df_filtered = df_filtered[~mask_debt_ratio_bad_data]

    print('\n shape after filtering:', df_filtered.shape, '\n')

    # seperate features and target
    X = df_filtered.drop(columns=['dlq_2yrs'])
    y = df_filtered['dlq_2yrs']

    # Do initial split, to seperate into training data, as well as validation data
    X_train_all, X_validate, y_train_all, y_validate = train_test_split(X, y, random_state=random_seed, train_size=train_size)

    # and split training data, to have small subset remainng for decision threshold tuning
    X_train, X_tune, y_train, y_tune = train_test_split(X_train_all, y_train_all, random_state=random_seed, train_size=train_prop_no_tune)


    # compute proportion of training, tuning, and validation samples that is positive class
    # can just compute average here as either 0 or 1. make into small dictionary
    sample_rate_dict = {'training': y_train.mean(), 
                        'tuning': y_tune.mean(),
                        'validation': y_validate.mean()}

    #TODO: do cleaner way to log data version, so that can configure which to use
    # create directory for logs of data version if not already created
    os.makedirs(LOGS_PATH / 'data' / '_v1', exist_ok=True)

    # dump class balance metrics to data log
    with open(LOGS_PATH / 'data' / '_v1' / 'data_balance.json', 'w') as f:
        json.dump(sample_rate_dict, f)

    # save data to pkl files
    with open(DATA_PATH / 'processed' / 'X_train.pkl', 'wb') as f:
        pkl.dump(X_train, f)

    with open(DATA_PATH / 'processed' / 'y_train.pkl', 'wb') as f:
        pkl.dump(y_train, f)
    
    with open(DATA_PATH / 'processed' / 'X_tune.pkl', 'wb') as f:
        pkl.dump(X_tune, f)

    with open(DATA_PATH / 'processed' / 'y_tune.pkl', 'wb') as f:
        pkl.dump(y_tune, f)

    with open(DATA_PATH / 'processed' / 'X_validate.pkl', 'wb') as f:
        pkl.dump(X_validate, f)

    with open(DATA_PATH / 'processed' / 'y_validate.pkl', 'wb') as f:
        pkl.dump(y_validate, f)


# if script run, then train the models!
if __name__ == '__main__':
    # if file called, then use default configuration path
    ingest_and_clean_data(config_path=CONFIG_PATH)