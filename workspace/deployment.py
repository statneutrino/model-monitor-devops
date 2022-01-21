from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copyfile


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
model_dir = os.path.join(config['output_model_path'])
ingest_dir = os.path.join(config['output_folder_path'])

####################function for deployment
def store_model_into_pickle(model_dir, ingest_dir, dest_dir):
    """#copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    Your model deployment function will not create new files; it will only copy existing files. It will copy your trained 
    model (trainedmodel.pkl), your model score (latestscore.txt), and a record of your ingested data (ingestedfiles.txt). 
    It will copy all three of these files from their original locations to a production deployment directory. 
    The location of the production deployment directory is specified in the prod_deployment_path entry 
    of your config.json starter file.
    """
    model_file = os.path.join(os.getcwd(), model_dir, 'trainedmodel.pkl')
    latest_score_file = os.path.join(os.getcwd(), model_dir, 'latestscore.txt')
    ingest_file = os.path.join(os.getcwd(), ingest_dir, 'ingestedfiles.txt')
    copyfile(model_file, os.path.join(os.getcwd(), dest_dir, 'trainedmodel.pkl'))
    copyfile(latest_score_file, os.path.join(os.getcwd(), dest_dir, 'latestscore.txt'))
    copyfile(ingest_file, os.path.join(os.getcwd(), dest_dir, 'ingestedfiles.txt'))


if __name__ == '__main__':
    store_model_into_pickle(model_dir, ingest_dir, prod_deployment_path)
    

