""" API that takes data and returns simple true or false prediction for default or not """

from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.schemas import features, prediction
from src.inference import return_inference
from src.paths import MODELS_PATH, CONFIG_PATH
import pickle as pkl
import yaml


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


# code to deal with application start up and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load in model at start up
    app.state.model = load_production_model()
    yield
    # clean up ML model and release the resources
    del app.state.model


# set up instance of API class
app = FastAPI(lifespan=lifespan)


# set up function to process POST request to 
@app.post('/predict')
def return_prediction(data: features):
    # run ML model
    inference, probability_default = return_inference(data, app.state.model)

    # return inference, using pydantic output schema
    return prediction(**{'inference': inference, 
                       'probability': probability_default})


# health endpoint to check that the API is running
@app.get('/health')
def health():
    return {'status': 'OK'}
