""" API that takes data and returns simple true or false prediction for default or not """

from fastapi import FastAPI
from src.schemas import features, prediction
from src.inference import return_inference
from src.paths import MODELS_PATH
import pickle as pkl

# set up instance of API class
app = FastAPI()

# load in model as a global variable
with open(MODELS_PATH / 'xgb.pkl', 'rb') as f:
        xgb_model = pkl.load(f)


# set up function to process POST request to 
@app.post('/predict')
def return_prediction(data: features):
    # run ML model
    inference = return_inference(data, xgb_model)
    # return inference, using pydantic class to inform schema
    return prediction(inference=inference)


# health endpoint to check that the API is running
@app.get('/health')
def health():
    return {'status': 'OK'}
