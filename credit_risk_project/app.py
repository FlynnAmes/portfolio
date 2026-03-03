""" API that takes data and returns simple true or false prediction for default or not """

from fastapi import FastAPI
from schemas import features, prediction
from inference import return_inference


# set up instance of API class
app = FastAPI()

# set up a test get function to sanity check that the API works
@app.get('/')
def get_a_signal():
    return {'msg': 'hello world!'}

# set up function to process POST request to 
@app.post('/predict')
def return_prediction(data: features):
    # run ML model
    inference = return_inference(data)
    # return inference, using pydantic class to inform schema
    return prediction(inference=inference)

#TODO: add health and ready endpoints. double check model loading