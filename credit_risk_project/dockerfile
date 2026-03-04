# initial dockerfile for repo, use to initiate app upon running the container
# first get the base image. 
FROM python:3.11-slim

# set working directory to project dir for now (will create seperate folder for app.py in future)
WORKDIR /app

# copy requirements file from host to container
COPY requirements.txt .

# set up pip package manager and install required dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# now copy remaining files to working directory
COPY . .

# start the application inside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]