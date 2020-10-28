# getting base image from python 3.7
FROM python:3.7.8

MAINTAINER rae wallace <raedjamw@gmail.com>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src/app

# install project requirements
RUN pip install --no-cache-dir gunicorn pandas numpy Flask Flask-Cors sklearn
RUN pip install --no-cache-dir keras==2.4.3
RUN pip install --no-cache-dir tensorflow==2.3.1


#Copy project
COPY . .

# functions Dir
WORKDIR /src/app/Shape_Functs_And_Tests
# Run functions
RUN python Dataframe_shape_functions.py

# Unit Test Dataframe shape functions
WORKDIR /src/app/Shape_Functs_And_Tests
RUN python Test_Dataframe_shape_functions.py

# Run Model and Generate H5 file
WORKDIR /src/app
RUN python BS_Model.py

# Unit Test API
WORKDIR /src/app
RUN python API_Test.py

# set work directory
WORKDIR /src/app

# set app port
EXPOSE 8020

# Run app.py when the container launches
CMD [ "gunicorn", "--bind", "0.0.0.0:8020", "app:app" ]
