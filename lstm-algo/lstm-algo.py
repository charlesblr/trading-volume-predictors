import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import tensorflow as tf

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

print(X[2], y[2])
print(X[3], y[3])




