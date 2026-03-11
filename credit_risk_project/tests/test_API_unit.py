""" unit tests for testing API """

import pytest
from src.app import app
from fastapi.testclient import TestClient

@pytest.mark.unit
def test_health_endpoint_correct_output():
    with TestClient(app) as client:
        assert dict(client.get('/health').json()) == {'status': 'OK'}


@pytest.mark.unit
def test_get_OK_status_code_good_data(input_data):

    with TestClient(app) as client:
        assert client.post('/predict', json=input_data).status_code == 200


@pytest.mark.unit
def test_get_expected_output_keys_with_good_data(input_data):

    with TestClient(app) as client:
        output = client.post('/predict', json=input_data).json()
        assert list(output.keys()) == ['decision', 'probability', 'decision_threshold']


@pytest.mark.unit
def test_get_error_status_code_bad_data(input_data):
    # test removing a row (using copy of input data)
    incomplete_data = input_data.copy()
    incomplete_data.pop('late_60_89')

    # should get error saying that data syntactically correct but could not be processed
    with TestClient(app) as client:
        assert client.post('/predict', json=incomplete_data).status_code == 422