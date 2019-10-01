# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:41:43 2019

@author: Saiprasad
"""

from math import sqrt
import tensorflow
import pandas as pd
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
 

dataset = pd.read_csv('Spatial_temporal_detection_trainset.csv')
X_com = dataset.iloc[:17850,1:].values

X_pro = np.zeros((8,17800,50))
y_pro = np.zeros((17800,8))

sc_X = MinMaxScaler()
X_com = sc_X.fit_transform(X_com)


for j in range(8):
    X = X_com[:,j]
    X = X.reshape(17850,1)    
    x_train = []
    y_train = []
    for i in range(50,17850):
        x_train.append(X[i-50:i,0])
        y_train.append(X[i,0])
    x_train = np.array(x_train)
    x_train = x_train.reshape(17800,50)
    y_train = np.array(y_train)
    y_train = y_train.reshape(17800,)
    X_pro[j,:,:] = x_train
    y_pro[:,j] = y_train
      
y_ans = np.zeros((17800,8))  
    
    
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

classifier = Sequential()

classifier.add(LSTM(100,input_shape=(50,1),return_sequences=True, activation='tanh'))
classifier.add(Dropout(0.2))

classifier.add(LSTM(100,return_sequences=True, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(LSTM(100,return_sequences=False, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(Dense(1))

classifier.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])

classifier.fit(x1, y1, batch_size = 30, epochs=100, validation_split = 0.2)

x1 = X_pro[0,:,:]
x1 = x1.reshape((17800,50,1))
y1 = y_pro[:,0]





 
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
 
# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
