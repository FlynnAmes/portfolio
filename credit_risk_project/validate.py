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

# first make sure that path is the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# load in validation data
with open('data/processed/X_validate.pkl', 'rb') as f:
    X_validate = pkl.load(f)
with open('data/processed/y_validate.pkl', 'rb') as f:
    y_validate = pkl.load(f)

# create an initial empty JSON file for the inference times of all models
Path('logs/inference_times.json').touch()
# add braces so that can append to file
with open('logs/inference_times.json', 'w') as f:
    f.write('{}')

# loop through all models
for i, model_path in enumerate(glob.glob('models/*.pkl')):
  
    with open(model_path, 'rb') as f:
        # get model
        model = pkl.load(f)

        # and its name (removing pkl extension and dir name)
        model_name = model_path.replace('.pkl', '').replace('models\\', '')
    
    print(f'model {model_name} loaded \n')

    ##########
    # call predict method for model to get predicted delinquency
    y_pred = model.predict(X_validate)
    print(f'prediction completed')

    # get inference time on validation data - here doing in batch rather than individually
    # as should be good proxy for relative difference between models
    inference_time = timeit.timeit(lambda: model.predict(X_validate), number=5)

    # get scoring metrics (including recall)
    scoring_metrics = classification_report(y_true=y_validate, y_pred=y_pred, output_dict=True)

    # get classification report for the model
    print(f'\n for the {model_name} model: \n', pd.DataFrame(scoring_metrics))

    # log scoring metrics
    current_date = datetime.now().strftime('%Y%m%d')

    # get name for run, using current time, as well as model name
    run_name = datetime.now().strftime('%Y%m%d') + f'_{model_name}'  # could use _%H%m%S on top
    savedir = 'logs/' + run_name

    # if dir name does not exist, make it
    if os.path.exists(savedir) is False:
        os.makedirs(savedir)
    
    # save scoring metrics
    with open(savedir + '/scoring_metrics', 'w') as f:
        json.dump(scoring_metrics, f, indent=4)
    
    # save inference time
    # first open file
    with open('logs/inference_times.json', 'r') as f:
        inference_file_data = json.load(f)
    
    # assign value
    inference_file_data[f'{model_name}'] = inference_time
    # and save (need to seperate owing to need for read and write permissions)
    with open('logs/inference_times.json', 'w') as f:
        json.dump(inference_file_data, f, indent=4)