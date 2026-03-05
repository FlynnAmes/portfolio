<h1> Overview </h1>

This project (polishing/refactoring in progress) provides a classifier that predicts whether or not someone will default on a loan within two years of acquiring it.
It is designed to be used by a (hypothetical) fintech company that provides a fast response 
to a loan application. For now the model is provides a simple 'accept or reject' output (where 1 indicates a possible delinquency and thus rejection). <br> <br>
The app is containerised via docker and tested using pytest (automated on push to repo via GitHub actions). Data is preprocessed with Pandas and NumPy. ML models are built using sklearn pipelines and XGboost. The application and its endpoints are configured using FastAPI.


<h2> Contents list </h2>

The project root directory currently contains:

<ul>

- <b> <i> data/ </i> </b> - raw data (see below for source) used to train model, as well as processed (i.e., cleaned) data used for model training and validation. <br>
- <b> <i> logs/ </i> </b> - upon running the code, contains logged scoring metrics for each trained model, along with cross-validation results and inference times (the latter computed during validation).<br>
- <b> <i> models/ </i> </b> - the trained ML models in pickle format. <br>
- <b> <i> notebooks_and_in_prog/ </i> </b> - notebooks for messier EDA and initial exploratory testing (note these are messy and currently lack much expalantion, will either delete or refine later). <br>
- <b> <i> src/ </i> </b> - Python scripts used for creating ML models and deploying the application. Includes initial cleaning of data, training of ML models, validation, inference, paths, schemas (pydantic classes for input and output data) and the application file itself. <br>
- <b> <i> tests/ </i> </b> - contains both unit and integration tests. <br>
- <b> <i> compose.yml </i> </b> - script that sets up running instance of a docker container, for running the code across any system. <br>
- <b> <i> config.yml </i> </b> - parameters used during model training (e.g., random seeds, number of folds in cross validation etc.). <br>
- <b> <i> dockerfile </i> </b> - used to create the docker image for running the project in an isolated and reproducible envrionment. <br>
- <b> <i> pyproject.toml </i> </b> - specifies custom 'marks' for pytests, and also build options (where building project as a package locally). <br>
- <b> <i> requirements.txt </i> </b> - contains python packages required to run the code. <br>

</ul>

<h2> Implementation details to note: </h2>

<ul>
<li> The model performs a single inference (rather than batch inference). This is to simulate an instant response to the loan application (e.g., on a fintech app). Therefore only one value must be given for each feature. </li>
<li> 10 input features are required. Their names and types can be found in src/schemas.py. </li>

</ul>


<h2> The data </h2>

The model is trained on a credit risk dataset linked <a href=https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset> here </a>


<h2> Model details </h2>

The algorithm employed in the inference model is an XGboost classifier.
Logistic regression models are also trained and their performance evaluated.

Recall of positive classifications (i.e., the proportion of people who defaulted who are correctly 
classified as doing so) is the primary metric used to choose the ML algorithm. This is because a false negative (i.e., not catching a potential delinquency) is deemed more costly than false positives.

Model inference time is also considered. Logistic regression outperforms XGboost here, but the performance of either algorithm is deemed good enough for the context (singular inferences obtainable in under a second).
Logistic regression interpretability is found to degrade with attempts to better tailor it 
to the non-linear decision boundary (Improved recall of positive classifications is accompanied by a reversal-in-sign in relationship between features and the log-of-odds of defaulting). Given these results, XGboost is chosen.

All models are trained using sklearn pipeline objects, to avoid data leakage. This also means that at inference, the model simply needs to be loaded and .predict() method called.
A randomSearchCV is used to obtain optimal hyperparameters (see src/train.py).

<h2> How to use the model and app to get a prediction </h2>

To guarantee the code/application always works on your machine, run it in a containerised environment via docker. First, <a href=https://docs.docker.com/desktop/> install docker </a> on your system.

Next, open a terminal, navigate to the project directory, and run the following:

``` 
docker compose up
``` 

This will first build a docker image from the dockerfile and then run an instance of that image (i.e., a container) on your system. Upon start-up of the container, the FastAPI server is launched.

Now the server is running on your machine, a POST HTTP request can be sent with feature data attached. Upon doing so, a prediction for defaulting is returned in JSON format. 
This can all be done via the requests package e.g., using the following:

``` 
import requests
response = requests.post('http://your_server_address:8000/predict', json=feature_dict)
print(response.content) 
```
where <i>feature_dict</i> is a dictionary of features (the keys) and their associated values and 
<i>your_server_address</i> is the address of the running instance of the server.
If you're running on your local machine (local host), this will be 127.0.0.1.

To run the code that trains the model (and logs output), perform the following command:

```
docker exec -i credit_risk_container python train.py
```

which runs the training code inside the container. Other files can be run by replacing train.py with the file of choice.

Once finished with the project, running instances of the container can be dismantled using:

```
docker compose down
```

<h2> Notes on running without docker </h2>

For best results without docker, one should create a seperate python environment. This project uses Python version 3.11. One can create a conda environment specifying the python version using the following command:

``` 
conda create -n <env_name> python=3.11
```

where <i>env_name</i> is an environment name you specify. Activate the environment:

```
conda activate <env_name>
```

Then navigate into the project directory and install the required python dependencies:

```
pip install -r requirements.txt
```

If running without docker, the project directory may need to be added to your python path otherwise relative imports may fail. This can be done manually, but a cleaner option is to install the project as a module:

```
pip install -e .
```

Files can then be run by running them as modules in the terminal, e.g., with the following command:

```
python -m src.train
```

which will run the training script (replace train with the file of choice) to train the ML models. The application server can be initiated with the following command:

```
uvicorn src.app:app
```