""" This file takes input data from API (in JSON form, assuming individual inference rather than batch),
and for now just returns the binary prediction about whether delinquent or not """

import os
import pickle as pkl
import pandas as pd
import numpy as np
from schemas import features
from paths import MODELS_PATH


#NOTE: will probably put this function elsewhere at some point. Need to figure out how to connect to the API

def return_inference(input_features):
    """ Function that takes in dictionary containing input data (would be converted from JSON after 
    receiveing through the API) and returns 
     simple 1 or 0 prediction for whether for delinquency, which the 
      'app' will then be able to use to decide whether to quick-accept/reject """
    
    #TODO: should convert type of incoming data to numpy etc. (because will be in standrad python types 
    # such as float and int initially)

    # test whether or not input actually is JSON format (NOTE: for now just raise exception, 
    # will sort graceful failure later/use pydantic along with API)
    if not isinstance(input_features, features):
        raise TypeError(f'input data file should be in a pydantic features format. It is currently a {type(input_features)}')

    # convert input pydantic baseModel to dictionary
    feature_dict = input_features.model_dump()

    # all columsn required to be in input dictionary (TODO: use saved column names from training in future)
    required_cols = set(('monthly_inc', 'rev_util', 'debt_ratio', 'open_credit',
                        'real_estate', 'late_30_59', 'late_60_89', 'late_90', 'dependents', 'age'))
    
    # test that all required columns in 
    if set((feature_dict.keys())) != required_cols:
        # .difference here returns columns that required but not in input dictionary
        raise KeyError(f'missing required columns in input dictionary: {required_cols.difference(feature_dict)}')

    with open(MODELS_PATH / 'xgb.pkl', 'rb') as f:
        # now load in the xgboost model
        model = pkl.load(f)

    # convert input dictionary to dataFrame. wrap in list because of scalar values (as assuming 
    # singular inference here)
    data_df = pd.DataFrame.from_dict([feature_dict])
 
    # REORDER columns to be same as used in model (because xgboost converts to arrays internally, 
    # so ordering of feaftures matter). Use .loc method for this
    data_df_reordered = data_df.loc[:, model.feature_names_in_]

    # then get predictions using the data (pipeline object takes care of preprocessing)
    # squeeze prediction so that 1D (because for now just want one prediction) - otherwise error
    # in newer python versions
    y_pred = np.squeeze(model.predict(data_df_reordered))

    # convert prediction to standard integer so that can be serialised by FastAPI
    return int(y_pred)

