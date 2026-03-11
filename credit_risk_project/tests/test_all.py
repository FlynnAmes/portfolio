""" Tests for input and output schemas as well as API, using pytest """

import pytest
from src.schemas import features, prediction
from pydantic import ValidationError
from src.app import app
from fastapi.testclient import TestClient
import requests

##########
# fixtures
##########

# good data with appropriate types (in different order to tat which model trained, but should 
# not matter)
@pytest.fixture
def input_data():
    return { 'rev_util': float(0.2),
             'age': int(37),
             'late_30_59': int(1), 
             'debt_ratio': float(0.2), 
             'open_credit': int(1),
             'late_90': int(0), 
             'dependents': int(2), 
             'real_estate': int(0),  
             'late_60_89': int(1), 
             'monthly_inc': float(2000),}

@pytest.fixture
def output_data():
    return {'inference': 1,
            'probability': 0.77}

######################
# define tests
######################

@pytest.mark.unit
def test_pydantic_feature_schema_accepts_good_data(input_data):
    assert features(**input_data)


@pytest.mark.unit
def test_pydantic_feature_schema_rejects_bad_data(input_data):
    # create copy of initial dictionary (pytest unpacks fixture by this point?)
    bad_input_data = input_data.copy()
    # alter one value of the good data to the wrong type
    bad_input_data['late_60_89'] = "a string"
    # then check whether appropiate error is raised
    with pytest.raises(ValidationError):
        features(**bad_input_data)


@pytest.mark.unit
# just accepts one value so no need for fixture yet
def test_pydantic_prediction_schema_accepts_good_data(output_data):
    assert prediction(**output_data)


@pytest.mark.unit
def test_pydantic_prediction_schema_rejects_bad_data(output_data):
    # change one of the values to make incorrect type
    bad_output_data = output_data.copy()
    bad_output_data['inference'] = 'a string'

    # then check whether appropiate error is raised
    with pytest.raises(ValidationError):
        features(**bad_output_data)

#################
# unit tests for API
#################

# # can set up testclient to test app without manually starting server
# client = TestClient(app)


@pytest.mark.unit
def test_get_OK_status_code_good_data(input_data):

    with TestClient(app) as client:
        assert client.post('/predict', json=input_data).status_code == 200


@pytest.mark.unit
def test_get_error_status_code_bad_data(input_data):
    # test removing a row
    incomplete_data = input_data.copy().pop('late_60_89')
    # should get error saying that data syntactically correct but could not be processed
    with TestClient(app) as client:
        assert client.post('/predict', json=incomplete_data).status_code == 422

##################
# integration tests for the API
##################

# test that the API is running after setup with docker
@pytest.mark.integration
def test_API_is_running_with_docker():
    assert requests.get('http://127.0.0.1:8000/health').status_code == 200


# test can get a prediction using valid input data when using docker
@pytest.mark.integration
def test_get_prediction_with_docker(input_data):
    assert requests.post('http://127.0.0.1:8000/predict', json=input_data).status_code == 200

