""" API that takes data and returns simple true or false prediction for default or not """

from fastapi import FastAPI
from fastapi import HTTPException
from contextlib import asynccontextmanager
from src.schemas import features, prediction
from src.inference import return_inference
from src.paths import MODELS_PATH, CONFIG_PATH, LOGS_PATH
import pickle as pkl
import os
import yaml
import logging
import json


def load_production_model():
    """ function to load model during app startup, given config params specifying 
    which model to load """

    # load in configuration parameters
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # get model type and decision threshold type to employ in production
    model_type = config['production_model_type']
    threshold_type = config['production_threshold_type']

    # then load in and return specified classifier object
    with open(MODELS_PATH / 'tuned' / model_type / (threshold_type + '.pkl'), 'rb') as f:
        return pkl.load(f)


def setup_logger():
    """ create logger used to log inference outputs """
    # set up logger for logging of inferences
    logger = logging.getLogger(__name__)
    # make logs path if no exist
    logs_dir = LOGS_PATH / 'inference'
    os.makedirs(logs_dir, exist_ok=True)
    # set handler to send logs to file - for now just simply send to one file
    # (not worrying about changing for different intervals/upon recahing memory limit)
    FileHandler = logging.FileHandler(logs_dir / 'app.log', mode='a')
    # add handler to logger
    logger.addHandler(FileHandler)
    # set formatter for logging
    formatter = logging.Formatter(fmt='{asctime} - {message}',
                        style='{',
                        datefmt='%Y%m%d %H%M')
    # and give to handler
    FileHandler.setFormatter(formatter)

    # set level to lowest for logger
    logger.setLevel('DEBUG')

    return logger


# code to deal with application start up and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load in model at start up
    app.state.model = load_production_model()
    # set up logger
    app.state.logger = setup_logger()
    yield

    # clean up and release the resources
    del app.state.model
    del app.state.logger


# set up instance of API class
app = FastAPI(lifespan=lifespan)


# set up function to process POST request to 
@app.post('/predict')
def return_prediction(data: features):
    try:
        # run ML model
        decision, probability_default, decision_threshold = return_inference(data, app.state.model)
    except Exception as e:
        # if inference fails for whatever reason, log it
        app.state.logger.error(f'inference failed; error: {str(e)}')
        # and return a more helpful error message
        raise HTTPException(status_code=500, detail='inference failed')

    # collate output to log
    output_to_log = json.dumps({
        'input': data.model_dump(),
        'decision': decision,
        'prob of default': probability_default,
        'decision_threshold': decision_threshold})

    # and log for the inference
    app.state.logger.info(f'prediction_made; info: {output_to_log}')

    # return inference, using pydantic output schema
    return prediction(**{'decision': decision, 
                       'probability_default': probability_default,
                       'decision_threshold': decision_threshold})


# health endpoint to check that the API is running
@app.get('/health')
def health():
    return {'status': 'OK'}


# ready enpoint to check that the server is running and ready to give output
@app.get('/ready')
def ready():
    # raise error if server is up but nodel not yet loaded
    if app.state.model is None:
        raise HTTPException(status=503, detail='model not yet loaded')
    else:
        return {'status': 'ready'}
