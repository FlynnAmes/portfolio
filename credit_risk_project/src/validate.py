""" Validate ML models using the remaining data, compute classification scores 
(namely recall), and save these """

import os
from pathlib import Path
import glob
import pickle as pkl
from sklearn.metrics import classification_report
import pandas as pd
import timeit
from datetime import datetime
import json
from paths import LOGS_PATH, MODELS_PATH, DATA_PATH

##########
# functions
##########


def log_validation_params(y_pred, y_validate, model_name):
    """ logs scoring metrics for ML model by running classification report. Uses 
     current time and model name as dir name for the logs. 
     
     y_pred: prediction of target
     y_test: actual target values
     model_name: string for name of model (to be used in saving)  """
    
    # get scoring metrics (including recall)
    scoring_metrics = classification_report(y_true=y_validate, y_pred=y_pred, output_dict=True)

    # get name for run, using current time, as well as model name
    run_name = datetime.now().strftime('%Y%m%d') + f'_{model_name}'
    savedir = LOGS_PATH / run_name

    # if dir name does not exist, make it
    os.makedirs(savedir, exist_ok=True)
    
    # save scoring metrics
    with open(savedir / 'scoring_metrics.json', 'w') as f:
        json.dump(scoring_metrics, f, indent=4)
    

def compute_and_log_inference_time(clf, X_validate, model_name, number=5):
    """ computes inference time for classifer model and saved to 
     json file (that must be defined in advance)
    
    clf: fitted model classifier object
    X_validate: training data to predict on
    number: number of times to perform inference for timing """

    # get inference time on validation data - here doing in batch rather than individually
    # as should be good proxy for relative difference between models
    inference_time = timeit.timeit(lambda: clf.predict(X_validate), number=number)

    # save inference time. first open file
    with open(LOGS_PATH / 'inference_times.json', 'r') as f:
        inference_file_data = json.load(f)
    
    # assign value
    inference_file_data[f'{model_name}'] = inference_time

    # and save (need to seperate owing to need for read and write permissions)
    with open(LOGS_PATH / 'inference_times.json', 'w') as f:
        json.dump(inference_file_data, f, indent=4)


def validate_models():
    """ main function for validating the models """
    ##############
    # data and files
    ##############

    # load in validation data
    with open(DATA_PATH / 'processed' / 'X_validate.pkl', 'rb') as f:
        X_validate = pkl.load(f)
    with open(DATA_PATH / 'processed' / 'y_validate.pkl', 'rb') as f:
        y_validate = pkl.load(f)

    # create an initial empty JSON file for the inference times of all models
    Path(LOGS_PATH / 'inference_times.json').touch()

    # add braces so that can append to file
    with open(LOGS_PATH / 'inference_times.json', 'w') as f:
        f.write('{}')


    ##########
    # main validation loop
    ##########

    # loop through all models
    for i, model_path in enumerate(glob.glob(str(MODELS_PATH / '*.pkl'))):
    
        with open(model_path, 'rb') as f:
            # get model
            model = pkl.load(f)
            # and its name (stem gives name of end of path, omitting extension)
            model_name = Path(model_path).stem
        
        print(f'model {model_name} loaded \n')

        ##########
        # call predict method for model to get predicted delinquency
        y_pred = model.predict(X_validate)

        print(f'prediction completed', '\n')

        # log scoring metrics
        log_validation_params(y_pred=y_pred, y_validate=y_validate, model_name=model_name)
        
        # compute and log inference times for model
        compute_and_log_inference_time(clf=model, X_validate=X_validate, model_name=model_name)

        print('params logged', '\n')


# if script run, then validate the models!
if __name__ == '__main__':
    validate_models()