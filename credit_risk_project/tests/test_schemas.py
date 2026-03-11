""" Tests for input and output schemas using pytest """

import pytest
from src.schemas import features, prediction
from pydantic import ValidationError

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
        assert features(**bad_output_data)





