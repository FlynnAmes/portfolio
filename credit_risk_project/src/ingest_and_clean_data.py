""" Ingest and filter out bad data, save seperate validation and training data """

import yaml
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import pickle as pkl
from src.paths import CONFIG_PATH, DATA_PATH

# get random seed and training proportion for train test split
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

random_seed = config['random_seed']
train_size = config['train_size']

# load in data
df = pd.read_csv(DATA_PATH / 'raw' / 'Credit_Risk_Benchmark_Dataset.csv')

#############
# Filtering
#############

print('initial shape: \n', df.shape)

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

print('shape after filtering: \n', df_filtered.shape)

# seperate features and target
X = df_filtered.drop(columns=['dlq_2yrs'])
y = df_filtered['dlq_2yrs']

# Do initial split, to seperate into train/test data, as well as validation data
X_train, X_validate, y_train, y_validate = train_test_split(X, y, random_state=random_seed, train_size=train_size)

# save data to pkl files
with open(DATA_PATH / 'processed' / 'X_train.pkl', 'wb') as f:
    pkl.dump(X_train, f)

with open(DATA_PATH / 'processed' / 'y_train.pkl', 'wb') as f:
    pkl.dump(y_train, f)

with open(DATA_PATH / 'processed' / 'X_validate.pkl', 'wb') as f:
    pkl.dump(X_validate, f)

with open(DATA_PATH / 'processed' / 'y_validate.pkl', 'wb') as f:
    pkl.dump(y_validate, f)