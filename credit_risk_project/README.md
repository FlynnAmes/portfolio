<h1> Overview </h1>

This project (polishing/productionising in progress) provides a classifier that predicts whether or not someone 
will default on a loan within two years of acquiring it.
It is designed to be used by a (hypothetical) fintech company that provides an fast response 
upon whether an loan application is to be rejected or move forwards for further processing.


<h2> Contents </h2>

The project contents are currently:

<ul>

<b> <i> data/ </i> </b> - raw data (see below for source) used to train model, as well as processed (i.e., cleaned) data used for model training and validation <br> <br>
<b> <i> logs/ </i> </b> - contains logged scoring metrics for each trained model, along with cross-validation results, and inference times (the latter computed during validation) <br> <br>
<b> <i> models/ </i> </b> - the trained ML models, in pickle format <br> <br>
<b> <i> notebooks_and_in_prog/ </i> </b> - notebooks for messier EDA and initial exploratory testing (may delete later) <br> <br>
<b> <i> app.py </i> </b> - Contains the API (using FastAPI). By running the API and thus activating the server, feature data can be sent via a HTTP post request, and the prediction (for now kept simple with 1 for default, 0 for no default) returned <br> <br>
<b> <i> config.yaml </i> </b> - parameters used during model training (e.g., random seeds, number of folds in cross validation etc.) <br> <br>
<b> <i> inference.py </i> </b> - contains the code that runs when a HTTP POST request is sent to the app which, for now, simply returns 1 for default, and 0 for no default <br> <br>
<b> <i> ingest_and_clean_data.py </i> </b> - performs cleaning of data (namely removing bad data - see notebooks_and_in_prog/EDA.ipynb for exploration of this - albeit currently messy) <br> <br>
<b> <i> requirements.yml </i> </b> - contains python packages required to run the code <br> <br>
<b> <i> schemas.py </i> </b> - contains pydantic classes used by app for input feature data and outputted prediction. These ensure robust validation and conversion of data types during the HTTP request <br> <br>
<b> <i> train.py </i> </b> - trains, creates, and saves the ML models <br> <br>
<b> <i> validate.py </i> </b> - Validates the ML models, and computes and logs scoring metrics, using held out data not seen during training <br> <br>
 </ul>

<h2> Implemetation details to note: </h2>

<ul>

<li> The model performs a single inference (rather than batch inference). This is to simulate an instant response to the loan application (e.g., on a fintech app). Therefore only one value must be given for each feature </li>
<li> 10 input features are required. Their names and types can be found in schemas.py

</ul>


<h2> The data </h2>

The model is trained on a credit risk data linked <a href=https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset> here </a>


<h2> Model details </h2>

The algorithm employed in the inference model is an XGboost classifier.
Logistic regression models are also trained and their performance evaluated.

Recall of positive classifications (i.e., the proportion of people who defaulted who are correctly 
classified as doing so) is the primary metric used to choose the ML algorithm. This is because a false negative (i.e., not catching a potential delinquency) is deemed more costly than false positives.

Model inference time is also considered. Logistic regression outperforms XGboost here, but the performance of either algorithm is deemed good enough for the context (singular inferences obtainable in under a second).
Logistic regression interpretability is found to degrade with attempts to better tailor it 
to the non-linear decision boundary (Improved recall of positive classifications is accompanied by a reversal-in-sign in relationship between features and the log-of-odds of defaulting). Given these results, XGboost is chosen.

All models are trained using sklearn pipeline objects, to avoid data leakage. This also means that at inference,
the model simply needs to be loaded and .predict() method called.
A randomSearchCV is used to obtain optimal hyperparameters (see train.py).

<h2> How to use the model and app to get a prediction </h2>

First create an environment containing the required python packages. Navigate to desired location in your terminal of choice and run:

``` 
conda env create --file requirements.yml
``` 

Then enter the environment using:

``` 
conda env activate creditRiskEnv
``` 

Now navigate to the project directory on your system. The saved models are already provided, but if one wants to train them, this can be accomplished by running train.py:

``` 
python train.py
``` 

To get a prediciton using new data, start the API server by typing:

``` 
fastapi run .\app.py
``` 

Once the server is running on your machine, a POST HTTP request can be sent to the server with feature data attached. Upon doing so, a prediction for defaulting is returned in JSON format. 
This can all be done via the requests package e.g., using the following:

``` 
import requests
response = requests.post('http://127.0.0.1:8000/', json=feature_dict)
print(response.content) 
```

where feature_dict is a dictionary of features (the keys) and their associated values.