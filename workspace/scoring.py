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
model_path = os.path.join(os.getcwd(), config['output_model_path'], 'trainedmodel.pkl')
column_names = ast.literal_eval(config['column_names'])

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    pass #f1score=metrics.f1_score(predicted,y)


if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), config['output_model_path'], 'trainedmodel.pkl')