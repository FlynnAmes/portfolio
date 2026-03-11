""" integration tests for testing running, containerised version of API """

import pytest
import requests


# test that the API is running after setup with docker
@pytest.mark.integration
def test_API_is_running_with_docker():
    assert requests.get('http://127.0.0.1:8000/health').status_code == 200


# test can get a prediction using valid input data when using docker
@pytest.mark.integration
def test_get_prediction_with_docker(input_data):
    assert requests.post('http://127.0.0.1:8000/predict', json=input_data).status_code == 200