import pytest

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
    return {'decision': 1,
            'probability_default': 0.77,
            'decision_threshold': 0.4}