import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load raw data 

raw_data = np.loadtxt("../input-vectors.csv", delimiter=",")

# We want every 10 minutes to have information on the next 50 minutes 
# i.e. we have batch of 10 vectors and we want to predict the next vector

# Split training and testing data 

n_steps = 10

def split_sequence(raw_data, n_steps):
	X, y = list(), list()
	for i in range(len(raw_data)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(raw_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = raw_data[i:end_ix], raw_data[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
 
X, y = split_sequence(raw_data, n_steps)

proportion_train = 0.8

def split_train_test(X, y, proportion_train):
	length = len(X)
	split_index = int(length * proportion_train)
	X_train, y_train = X[:split_index], y[:split_index]
	X_test, y_test = X[split_index:], y[split_index:]
	return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = split_train_test(X, y, proportion_train)

# print(np.shape(X))
# [samples, timesteps, features] = (47847, 10, 25)

# LSTM model with Keras

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

n_features = 25
n_units = 50
 
model = Sequential()

model.add(LSTM(n_units, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(n_features))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=2, verbose=2)

# Check prediction 

y_hat = model.predict(X_test, verbose=2)

print(y_hat)

