## Overview


This project implements an end-to-end machine learning system for predicting
loan default risk. 
The system includes:

- Data preprocessing and feature pipelines (Pandas + NumPy + scikit-learn + XGBoost)
- Model training and hyperparameter tuning (RandomSearchCV)
- Model validation and decision threshold optimisation
- A production inference API (FastAPI)
- Containerised deployment (Docker)
- Automated testing and CI (Pytest + GitHub Actions)


The API returns:
- predicted default class
- probability of default
- decision threshold used for classification


The system simulates a real-time credit decision engine for a fintech
loan application workflow.


## Repository Structure

```
├── data/
│   ├── processed/
│   └── raw/   
│  
├── logs/                          # training, validation, and inference logs (gitignored)
├── models/                        # trained model artifacts (gitignored)
├── notebooks/                     # exploratory analysis
│
├── src/                           # training, validation, and inference code
│   ├── app.py
│   ├── inference.py
│   ├── ingest_and_clean_data.py
│   ├── paths.py
│   ├── schemas.py
│   ├── train.py
│   ├── tune.py
│   └── validate.py
│   
├── tests/                         # unit and integration tests
│
├── Dockerfile
├── compose.yml
├── config.yml
├── pyproject.toml
├── requirements.txt
├── .gitignore
└──  README.md
```


Models, processed data, and logs are excluded from version control. <br>
All artifacts can be regenerated using the training pipeline.

## Data

The model is trained on a credit risk dataset linked <a href=https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset> here</a>. 


Note the dataset has an artificial class balance of 50% defaulting. This is >10 times larger than the proportion of defaults typically observed in a credit risk population.
Therefore, probabilities of default, obtained during inference, should not be interpreted as real world default probabilities (expected to be much lower using real world data). <br>
While a correction to outputted probabilities can be made to account for sample vs population class balance discrepancies, robust calibration of these probabilities would require a large dataset (because the limited prevalence of positive classifications would result in very large confidence intervals at larger probabilities).

The brier scores and reliability diagrams are therefore computed assuming that the population encountered during 
inference has the same class balance as that used in training.


## Model

**Models evaluated:** Logistic Regression and XGBoost  
**Model selection:** RandomSearchCV using average precision  
**Threshold optimisation:** F-beta score with recall weighted higher than precision  
**Final model:** XGBoost classifier (configurable variants)  
**Inference latency:** < 1 second per prediction


## Quickstart 

First download the data <a href=https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset> here</a> (placing in data/raw) and <a href=https://docs.docker.com/desktop/> install docker </a>if you haven't already.

With a docker server up and running, open a terminal, navigate to the repository directory, and use the following command:

``` 
docker compose up -d
``` 

This will build the docker image and run a corresponding container, within which the API server will be launched.

After this, a POST HTTP request can be sent to the server with feature data attached, e.g., using the requests package in python:

``` 
import requests
response = requests.post('http://your_server_address:8000/predict', json=feature_dict)
```

where <i>feature_dict</i> contains 10 input features validated using Pydantic schemas (see src/schemas.py). <br>
The response from the API will look something like:


```
{
  "prediction": 1,
  "probability_default": 0.77,
  "decision_threshold": 0.55
}
```

To run the code that trains the model (and logs output), perform the following command:

```
docker exec -i credit_risk_container python src/train.py
```

which runs the training code inside the container. Other files can be run by replacing train.py with the file of choice.

Once finished with the project, running instances of the container can be dismantled using:

```
docker compose down
```

## Running without Docker

In the repository directory, run the following commands in order to get the server running

```
pip install -r requirements.txt
pip install -e .
uvicorn src.app:app
```

## Future improvements

- Use inference logs for monitoring model performance and detecting data drift
- Model versioning and experimeny tracking using MLflow
- Simple pipeline orchestration using Prefect


