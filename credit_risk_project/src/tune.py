""" Produces model objects, with specified tuning of decision thresholds.
For now F_beta score is used to tune the model """

import os
from pathlib import Path
import yaml
import json
from src.paths import MODELS_PATH, CONFIG_PATH, DATA_PATH, LOGS_PATH
from glob import glob
import pickle as pkl
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV
from datetime import datetime
import numpy as np


def log_tuning_results(model_object, model_name: str):
    """ log the decision probability threshold of the tuned model object, as well as 
    the results of the tuning process. Save to dircteory marked by date. """

    # create save directory for logs if not already created. Fow now using day to 
    # distinguish between versions/experiments
    savedir = LOGS_PATH / model_name / datetime.now().strftime('%Y%m%D')
    os.makedirs(savedir, exist_ok=True)

    # create dictionary with decision threshold and its performance. convert to float to guarantee
    # serialisation (in case of XHBoost behaviour where can output float32)
    final_threshold_dict = {'threshold': float(model_object.best_threshold_),
                            'score': float(model_object.best_score_)}
    
    # get tuning results and convert arrays to list for serialisation (converting to float64 first 
    # to deal with xbgoost issue with float32)
    tuning_results = {k: list(np.array(var, dtype=np.float64)) for k, var in model_object.cv_results_.items()}
  
    # dump both to json
    with open(savedir / 'decision_threshold.json', 'w') as f:
        json.dump(final_threshold_dict, f, indent=4)
    
    with open(savedir / 'tuning_results.json', 'w') as f:
        json.dump(tuning_results, f, indent=4)
    
 


def save_model(model_object, model_name: str):
    """ save tuned model to pkl format, given name of model (string)"""

    with open(MODELS_PATH / 'tuned' / (model_name + '.pkl'), 'wb') as f:
            pkl.dump(model_object, f)


def tune_models(config_path):
    
    # load in tuning data (features and target)
    with open(DATA_PATH / 'processed' / 'X_tune.pkl', 'rb') as f:
        X_tune = pkl.load(f)
    
    with open(DATA_PATH / 'processed' / 'y_tune.pkl', 'rb') as f:
        y_tune = pkl.load(f)

    # now load in config params
    with open(config_path, 'r') as f:
        # extract dictionary containing weightings for recall to use in tuning models
        r_weightings = yaml.safe_load(f)['recall_weightings']

    # create save directory for tuned models if not already exist
    os.makedirs(MODELS_PATH / 'tuned', exist_ok=True)


    # load in model objects
    for model_path in glob(str(MODELS_PATH / 'pretuning' / '*.pkl')):

        # load the model
        with open(model_path, 'rb') as f:
            pretuned_model = pkl.load(f)

        # now loop over all recall weightings to create multiple tuned instances of the model
        for i in r_weightings.keys():
            
            # create model name using name of untuned model and key from r_weightings
            model_name = Path(model_path).stem + '_' + i

            # compute f_beta score using specified weighting
            f_beta_scorer = make_scorer(fbeta_score, beta=r_weightings[i])

            # perform tuning of model object using f beta score, given tuning data not seen during training
            # note that no refitting of the model is performed here
            clf_tuned = TunedThresholdClassifierCV(pretuned_model, scoring=f_beta_scorer, 
                                                            cv='prefit', refit=False, 
                                                            store_cv_results=True).fit(X_tune, y_tune)
            print('\n model tuned \n')

            # now save the tuned model
            save_model(model_object=clf_tuned,  model_name=model_name)

            print('\n model saved \n')

            # log tuning parameters (new decision trheshold + results from tuning tests)
            log_tuning_results(model_object=clf_tuned, model_name=model_name)

            print('\n tuning params logged \n')
        


if __name__ == '__main__':
    # if file called then use default configuration path
    tune_models(config_path=CONFIG_PATH)