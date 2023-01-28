from math import sqrt
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
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('Spatial_temporal_detection_trainset.csv', header=0, index_col=0)
values = dataset.values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 4, 1)

 
# split into train and test sets
values = reframed.values
n_train_hours = 12460
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, 0:32], train[:, 32:40]
test_X, test_y = test[:, 0:32], test[:, 32:40]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 4, 8))
test_X = test_X.reshape((test_X.shape[0], 4, 8))
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1]))
test_y = test_y.reshape((test_y.shape[0], test_y.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
 
# design network
from tensorflow.keras.layers import Dropout

classifier = Sequential()

classifier.add(LSTM(500,input_shape=(4,8),return_sequences=True, activation='tanh'))
classifier.add(Dropout(0.2))

classifier.add(LSTM(500,return_sequences=True, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(LSTM(500,return_sequences=True, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(LSTM(500,return_sequences=False, activation = 'tanh'))
classifier.add(Dropout(0.2))

classifier.add(Dense(100))
classifier.add(Dense(8))

classifier.compile(optimizer='adam', loss='mean_squared_error')
# fit network
history = classifier.fit(train_X, train_y, epochs=50, batch_size=72, validation_split=0.1, verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 08:33:39 2019

@author: Saiprasad
"""

