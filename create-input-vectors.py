import numpy as np
import pandas as pd

df_stock_data = pd.read_csv("US1.GILD_200101_200701.csv") # Starts at index 0

length = len(df_stock_data)

print(df_stock_data.head())
print(df_stock_data["<VOL>"].iloc[0])

# Creation of the input vectors 

columns_of_interest = ["<VOL>", "<HIGH>", "<LOW>", "<CLOSE>", "<OPEN>"]

W = np.zeros(shape=(length-50, 25))

for minute in range(0, length-50):
	for counter_col, col in enumerate(columns_of_interest):
		for counter_offset, offset in enumerate([0, 10, 20, 30, 40]):
			delta = df_stock_data[col].iloc[minute+10+offset] - df_stock_data[col].iloc[minute+offset]
			W[minute][counter_offset+counter_col*5] = delta

print(W[0])

np.savetxt("input-vectors.csv", W, delimiter=",")

# To load the input vectors do : W = np.loadtxt("input-vectors.csv", delimiter=",")