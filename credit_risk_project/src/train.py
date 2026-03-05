""" Train and track the performance of a logistic regression model , as well as an XGboost model, 
in predicting credit risk for a simple benchmark dataset """

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, FunctionTransformer, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import pickle as pkl
import yaml
import json
from datetime import datetime
from src.paths import LOGS_PATH, MODELS_PATH, CONFIG_PATH, DATA_PATH

###########
# functions
###########

def log_training_params(clf, model_name):
    """ Logs hyperparameters of sklearn estimators, as well as
      cross validation results obtained during training, and feature names for the XGboost
      Logs are saved using the current date, along with a name given to the model, to 
      the logs/ directory

      clf: the trained model pipeline RandomSearchCV object
      model_name: the name of the model (string), used for saving """
    
    pipeline_params = clf.best_estimator_.get_params()

    # then convert to string where not simple python object, so can serialise into json
    # (nested loop fine here as not many params)
    pipeline_params_reformat = {k: str(var) if not isinstance(var, (int, float, str, bool, type(None))) 
                                else var for k, var in pipeline_params.items()}

    # get name for run, using current time, as well as model name
    run_name = datetime.now().strftime('%Y%m%d') + f'_{model_name}'  # could use _%H%m%S on top
    savedir = LOGS_PATH / run_name

    # if dir name does not exist, make it
    os.makedirs(savedir, exist_ok=True)

    # and finally dump to JSON
    with open(savedir / 'hyperparams.json', 'w') as f:
        json.dump(pipeline_params_reformat, f, indent=4)

    # also dump to JSON the cv results (to check if large variation in this)
    pd.DataFrame(data=clf.cv_results_).to_csv(savedir / 'cv_results.csv')


def save_model(clf, model_name):
    """ saves the ML model to the models directory in pickle format, using model name (along 
     with MODELS PATH defined in paths.py) """

    with open(MODELS_PATH / (model_name + '.pkl'), 'wb') as f:
        pkl.dump(clf, f)


def train_models():
    """ main function for training the models """
    ###############
    # loading config parameters and data
    ###############

    # get from yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    random_seed = config['random_seed']
    spline_deg = config['spline_deg']
    spline_knots = config['spline_knots']
    cv_folds = config['cv_folds']
    cv_scoring_metric = config['cv_scoring_metric']
    lr_max_iter = config['logistic_regression_max_iter']

    # load in training data
    X_train = pd.read_pickle(DATA_PATH / 'processed' / 'X_train.pkl')
    y_train = pd.read_pickle(DATA_PATH / 'processed' / 'y_train.pkl')

    ###########
    # transformers and pipelines
    ###########

    # create log transformer
    LogTransformer = FunctionTransformer(np.log1p)

    # create custom columntransformer to only log scale those that need logging (for Logistic regression)
    ctLogScaleSome = ColumnTransformer([('scale_log', LogTransformer, ['monthly_inc', 'rev_util', 'debt_ratio'])], 
                                        remainder='passthrough')
    # create column transformer that applied spline with 3 knots to open credit (determined from EDA)
    ctLogSomeSplineOpCred = ColumnTransformer([('scale_log', LogTransformer, ['monthly_inc', 'rev_util', 'debt_ratio']), 
                                            ('spline', SplineTransformer(n_knots=spline_knots, degree=spline_deg), ['open_credit'])], 
                                        remainder='passthrough')

    # create pipes for logistic regression, firstly all log transformed, then only some, and finally some with open credit
    # a spline transform applied. saga solver to allow elastic net regression if needed.
    # Note all are standarised afterwards
    pipeLrAllLog = Pipeline([('scale_log', LogTransformer), ('scale_stand', StandardScaler()), 
                            ('clf', LogisticRegression(solver='saga', random_state=random_seed, max_iter=lr_max_iter))])
    pipeLrSomeLog = Pipeline([('scale_log', ctLogScaleSome), ('scale_stand', StandardScaler()), 
                            ('clf', LogisticRegression(solver='saga', random_state=random_seed, max_iter=lr_max_iter))])
    pipeLrSomeLogSomeSpline = Pipeline([('scale_log', ctLogSomeSplineOpCred), ('scale_stand', StandardScaler()), 
                            ('clf', LogisticRegression(solver='saga', random_state=random_seed, max_iter=lr_max_iter))])
    # and an xgboost - no need for scaling here
    pipe_xgb = Pipeline([('clf', XGBClassifier(random_seed=random_seed))])

    #############
    # parameter dictionaries
    #############

    # dictionary for parameters to search in random CV, as a test
    params_dict_lr = {
        'clf__l1_ratio': stats.uniform(),  # uniform distribution between 0 and 1
        'clf__C': stats.loguniform(1e-1, 1e1)  # log unform because order of magnitudes matter here
    }

    #TODO: test early stopping at some point
    params_dict_xgb = {'clf__learning_rate': stats.loguniform(0.01, 1), 
                            'clf__n_estimators': np.linspace(100, 2000, 20, dtype=np.int64),
                            'clf__max_depth': np.linspace(2, 20, 10, dtype=np.int64), 
                            'clf__gamma': np.linspace(0, 5, 6, dtype=np.float64),
                            'clf__subsample': np.linspace(0.1, 1, 10, dtype=np.float64),
                            'clf__colsample_bytree': np.linspace(0.1, 1, 10, dtype=np.float64),
                            'clf__reg_alpha': np.linspace(0, 1, 11, dtype=np.int64),
                            'clf__reg_lambda': np.linspace(0, 1, 11, dtype=np.int64),   
    }

    # set up dictionary containing models so can easily loop through the configs
    pipe_dictionary = {'lrAllLog': {'pipeline': pipeLrAllLog,
                                    'params_dict': params_dict_lr,
                                    },
                    'lrSomeLog': {'pipeline': pipeLrSomeLog,
                                    'params_dict': params_dict_lr,
                                    },
                    'lrSomeLogSomeSpline': {'pipeline': pipeLrSomeLogSomeSpline,
                                            'params_dict': params_dict_lr
                                            },
                    'xgb': {'pipeline': pipe_xgb,
                            'params_dict': params_dict_xgb,
                            },
                    }

    #######################
    # main training loop
    #######################

    for model in pipe_dictionary.keys():

        # set up randomised grid search, with corresponding pipeline and params dict

        clf = RandomizedSearchCV(pipe_dictionary[model]['pipeline'], pipe_dictionary[model]['params_dict'], 
                                cv=cv_folds, scoring=cv_scoring_metric, random_state=random_seed)
        
        print(model, '\n')

        # fit the classifier to training data. random search will test hyperparameters in different folds. 
        # Will then return model object at end fitted to entire training dataset
        clf.fit(X_train, y_train)

        print('model fitted', '\n')

        # then serialise model as pkl, saving it
        save_model(clf=clf, model_name=model)
        
        print('model saved', '\n')

        # now log parameters in model
        log_training_params(clf=clf, model_name=model)

        print('params logged', '\n')


# if script run, then train the models!
if __name__ == '__main__':
    train_models()
