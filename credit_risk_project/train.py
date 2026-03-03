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

# first make sure that path is the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# get constants from yaml
with open('config.yml') as f:
    config = yaml.safe_load(f)

random_seed = config['random_seed']
spline_deg = config['spline_deg']
spline_knots = config['spline_knots']
cv_folds = config['cv_folds']
cv_scoring_metric = config['cv_scoring_metric']

# load in training data
X_train = pd.read_pickle('data/processed/X_train.pkl')
y_train = pd.read_pickle('data/processed/y_train.pkl')

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
                         ('clf', LogisticRegression(solver='saga', random_state=random_seed, max_iter=1000))])
pipeLrSomeLog = Pipeline([('scale_log', ctLogScaleSome), ('scale_stand', StandardScaler()), 
                        ('clf', LogisticRegression(solver='saga', random_state=random_seed, max_iter=1000))])
pipeLrSomeLogSomeSpline = Pipeline([('scale_log', ctLogSomeSplineOpCred), ('scale_stand', StandardScaler()), 
                        ('clf', LogisticRegression(solver='saga', random_state=random_seed, max_iter=1000))])

# # and an xgboost - no need for scaling here
# pipe_xgb = Pipeline(('clf', XGBClassifier()))
# and an xgboost - no need for scaling here
pipe_xgb = Pipeline([('clf', XGBClassifier())])

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

    # # fit the classifier to training data. random search will test hyperparameters in different folds. 
    # # Will then return model object at end fitted to entire training dataset
    clf.fit(X_train, y_train)
    print('model fitted', '\n')

    # then serialise model as pkl, saving it
    with open(f'models/{model}.pkl', 'wb') as f:
        pkl.dump(clf, f)
    
    print('model saved', '\n')
    # dump model pipeline estimators and hyperparameter to JSON, first getting
    pipeline_params = clf.best_estimator_.get_params()

    # then convert to string where not simple python object, so can serialise into json
    # (nested loop fine here as not many params)
    pipeline_params_reformat = {k: str(var) if not isinstance(var, (int, float, str, bool, type(None))) 
                                else var for k, var in pipeline_params.items()}

    # get name for run, using current time, as well as model name
    run_name = datetime.now().strftime('%Y%m%d') + f'_{model}'  # could use _%H%m%S on top
    savedir = 'logs/' + run_name

    # if dir name does not exist, make it
    if os.path.exists(savedir) is False:
        os.makedirs(savedir)

    # and finally dump to JSON
    with open(savedir + '/hyperparams.json', 'w') as f:
        json.dump(pipeline_params_reformat, f, indent=4)

    # also dump to JSON the cv results (to check if large variation in this)
    pd.DataFrame(data=clf.cv_results_).to_csv(savedir + '/cv_results.csv')

    print('params logged', '\n')
