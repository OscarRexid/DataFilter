import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.linear_model import LinearRegression
import pyswarms as ps
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_excel("data.xlsx")
print(df)
df_filtered = df[(df > 0).all(axis=1)]
print(df_filtered)
data_array = np.array(df_filtered)
values, params = data_array[:,0], data_array[:,1:]

# Assuming params is your matrix of parameters and values is your function values
#params = np.random.rand(100, 2)  # 100 samples, each has 2 parameters
#values = np.random.rand(100)  # 100 function values

# Split the data into training and testing sets
params_train, params_test, values_train, values_test = train_test_split(params, values, test_size=0.2, random_state=42)

# Initialize a random forest regressor
rf = RandomForestRegressor(n_estimators=1500, random_state=42)

# Train the model on your data
rf.fit(params_train, values_train)

# Predict function values for the test set
values_pred = rf.predict(params_test)

# Calculate the mean squared error of the predictions
mse = r2_score(values_test, values_pred)
print(f'The Mean Squared Error of the predictions is {mse}')


def cost_func(position):
    return rf.predict(position)



options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
max = []
min = []
dim= params.shape[1]
for i in range(0,dim):
  max.append(np.amax(params[:,i]))  
  min.append(np.amin(params[:,i]))  
bounds = (min,max)
print(dim)
print(bounds)


optimizer = ps.single.GlobalBestPSO(n_particles=1000, dimensions=dim,
                                   options=options, bounds=bounds)

# Perform optimization
cost,pos = optimizer.optimize(cost_func, iters=1000)

print(cost)
print(pos)


