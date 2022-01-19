import pandas as pd
import pickle
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import ast
import numpy as np

# Load data
with open('l3final.pkl', 'rb') as file:
    model = pickle.load(file)

testdata=pd.read_csv('testdatafinal.csv')
print(testdata.head())

# Calculate MSE

X = testdata[['timeperiod']].values.reshape(-1,1)
y = testdata['sales'].values.reshape(-1,1)

predicted=model.predict(X)

mse=metrics.mean_squared_error(predicted,y)
print(mse)

# Conduct parametric significance test
with open('l3finalscores.txt', 'r') as f:
    mse_list = ast.literal_eval(f.read())

iqr = np.quantile(mse_list,0.75)-np.quantile(mse_list,0.25)
non_param_test = mse > np.quantile(mse_list,0.25)-iqr*1.5
print(iqr)
print(non_param_test)