# Std Imports
import pandas as pd
import numpy as np
import math
import random as rn
import os

# ML and DL Imports
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Activation, Dense
from keras.optimizers import SGD, Adam
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# User Defined Modules
import Shape_Functs_And_Tests.Dataframe_shape_functions as dfsf

# Random Seed to keep results consistent
rn.seed(1)
np.random.seed(1)
tf.random.set_seed(2)

# Path for saving to disk
save_path = "."

# Read in the dataframe
data = pd.read_csv("Model_H5/bitcoin.csv").drop(['time_period_start', 'time_period_end', 'time_open', 'time_close','Unnamed: 0'	], axis=1)
print(data.head(5))


# Run Lookback function
features = dfsf.lookback(data.drop(['price_open', 'price_low', 'price_close', 'volume_traded', 'trades_count'], axis=1), 60)


#Target Variable
target = features['price_high']
# Feature Variables
features = features.drop('price_high', axis=1)

# splitting data sets into train and test split
training_size = int(len(features)*0.75)
x_train,y_train = features.iloc[0:training_size,:], target.iloc[0:training_size]
x_test, y_test = features.iloc[training_size:len(features),:], target.iloc[training_size:len(features)]

# reshape input to be [samples, time steps, features] which is required for LSTM
x_train = x_train.values.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.values.reshape(x_test.shape[0],x_test.shape[1] , 1)


# fit an LSTM network to training data
def fit_lstm(Xt,yt,xts,yts,epoch, neurons):
  """
  This is LSTM RNN. It takes the train and validation sets,number of epochs
  and number of neurons as input. LSTM Layers are required for time series
  analysis

  """
  model = Sequential()
  # Relu in hidden because it works really well for regression
  model.add(LSTM(neurons, return_sequences=True, input_shape=(Xt.shape[1],Xt.shape[2]),activation='relu'))
  # # Relu in hidden to present against the vanishing gradient
  model.add(LSTM(neurons,activation='relu'))
  #Output layer default is Linear, this is a desirable activation function on regression problems.
  model.add(Dense(1))
  # MSE is the appropriate for regression,adam optimizer performs well in most cases
  model.compile(
    loss='mse',
    optimizer=Adam(lr=0.01),
    metrics=['mean_squared_error', 'mean_absolute_error']
  )
  history = model.fit(Xt,yt,validation_data=(xts,yts),epochs=epoch,verbose=1)

  return model

# initialize the fit_lstm function
lstm_model = fit_lstm(x_train,y_train,x_test,y_test,10,60)


# Lets Do the prediction and check performance metrics
train_predict = lstm_model.predict(x_train)
test_predict = lstm_model.predict(x_test)

# Calculate RMSE performance metrics
print('train RMSE',math.sqrt(mean_squared_error(y_train,train_predict)))
print('Test RMSE:',math.sqrt(mean_squared_error(y_test,test_predict)))

# save entire network to HDF5 (save everything, suggested)
lstm_model.save(os.path.join(save_path,"model_LSTM.h5"))
print("[INFO]: Finished saving model...")

