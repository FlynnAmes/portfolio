""" Defining the pydantic classes for input features and outputted predictions """

from pydantic import BaseModel

# define class for input data
# for now forcing variables that should be integers to be integers
class features(BaseModel):
    rev_util: float
    age: int
    late_30_59: int
    debt_ratio: float
    open_credit: int
    late_90: int
    # note american english here
    dependents: int
    real_estate: int
    late_60_89: int
    monthly_inc: float


class prediction(BaseModel):
    inference: int