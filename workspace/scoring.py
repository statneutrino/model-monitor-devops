from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import ast
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
target_name = config['target']
features = ast.literal_eval(config['feature_names'])

# Function for model scoring
def score_model(model_dir, test_data_path, features, target_name = 'exited', save = True):
    """#this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    """
    # Load data and reshape for fitting
    file_path = os.path.join(os.getcwd(), test_data_path)
    test_data = pd.read_csv(file_path)

    y = test_data[target_name].values.reshape(-1, 1).ravel()
    X = test_data.loc[:,features].values.reshape(-1, len(features))
    
    # Load trained model and calculate an F1 score with predictions
    with open(os.path.join(model_dir, 'trainedmodel.pkl'), 'rb') as file:
        clf = pickle.load(file)
    y_pred = clf.predict(X)
    f1score = metrics.f1_score(y_pred, y)

    if save is True:
        with open(os.path.join(model_dir, 'latestscore.txt'), 'w') as file:
            file.write(str(f1score))

    return f1score

if __name__ == '__main__':
    model_dir = os.path.join(os.getcwd(), config['output_model_path'])
    print(score_model(model_dir, test_data_path, features))