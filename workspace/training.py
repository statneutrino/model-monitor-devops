from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import ast

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
model_path = os.path.join(os.getcwd(), config['output_model_path'], 'trainedmodel.pkl')
target = config['target']
features = ast.literal_eval(config['feature_names'])

#################Function for training the model
def train_model(
    dataset_csv_path, 
    clf_path, random_state=42, 
    target_name='exited', 
    features=['lastmonth_activity', 'lastyear_activity', 'number_of_employees']):
    
    # Load data and reshape for fitting
    file_path = os.path.join(os.getcwd(), dataset_csv_path)
    ingested_data = pd.read_csv(file_path)
    
    y = ingested_data[target_name].values.reshape(-1, 1).ravel()
    X = ingested_data.loc[:,features].values.reshape(-1, len(features))
    
    print(X)
    # Use ridge regression logistic regression classifier for ML model
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='warn', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    clf = LogisticRegression(random_state=random_state).fit(X, y)
    
    # pickle trained classifier
    pickle.dump(clf, open(clf_path, 'wb'))

    return clf


if __name__ == '__main__':
    clf = train_model(dataset_csv_path, model_path, random_state=42, target_name='exited')
    
    # Test predictions
    file_path = os.path.join(os.getcwd(), dataset_csv_path)
    ingested_data = pd.read_csv(file_path)
    X = ingested_data.loc[:,features].values.reshape(-1, len(features))
    print(clf.predict(X))