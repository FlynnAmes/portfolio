""" This file takes input data from API (assuming pydantic baseModel subclass, see schemas.py),
 assuming individual inference rather than batch.
For now just returns the binary prediction about whether delinquent or not """

import pandas as pd
import numpy as np
from src.schemas import features


def return_inference(input_features, model_object):
    """ Main function that takes input feature data (a pydantic baseModel subclass object) and returns 
     simple 1 or 0 prediction for whether for delinquency, which the 
      'app' will then be able to use to decide whether to quick-accept/reject.
       
      other argument is the XGB model object """

    #NOTE: if doing batch inference may want to convert to numpy etc.?

    # test whether or not input actually is pydantic baseModel feature object
    if not isinstance(input_features, features):
        raise TypeError(f'input data file should be in a pydantic features format. It is currently a {type(input_features)}')

    # convert input pydantic baseModel to dictionary
    feature_dict = input_features.model_dump()

    # convert input dictionary to dataFrame. wrap in list because of scalar values (as assuming 
    # singular inference here)
    data_df = pd.DataFrame.from_dict([feature_dict])
 
    # REORDER columns to be same as used in model (because xgboost converts to arrays internally, 
    # so ordering of feaftures matter). Use .loc method for this
    data_df_reordered = data_df.loc[:, model_object.feature_names_in_]

    # then get predictions using the data (pipeline object takes care of preprocessing)
    # squeeze prediction so that 1D (because for now just want one prediction) - otherwise error
    # in newer python versions
    y_pred = np.squeeze(model_object.predict(data_df_reordered))

    # get predicted probability
    y_pred_proba_default = np.squeeze(model_object.predict_proba(data_df_reordered))[-1]


    # return the decision, probability of defaut and decision threshold used
    # convert numpy format to standard int and float so that can be serialised by FastAPI
    return int(y_pred), float(y_pred_proba_default), float(model_object.best_threshold_)