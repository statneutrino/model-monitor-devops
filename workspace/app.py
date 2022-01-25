from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, execution_time, missing_cols, outdated_packages_list
import json
import os
import ast


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 

deployed_model_dir = config['prod_deployment_path']
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 
features = ast.literal_eval(config['feature_names'])
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 

data_df = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path))

# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    #call the prediction function you created in Step 3
    input_data_path = request.args.get("inputdata")
    y_pred = model_predictions(deployed_model_path, input_data_path, features, target_name = 'exited')
    return str(y_pred)

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    f1_score = score_model(deployed_model_dir, test_data_path, features)
    return str(f1_score)

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    summary_stats = dataframe_summary(data_df)
    return str(summary_stats)


# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    diag_dict = {}
    # check timing and percent NA values
    ex_times = execution_time()
    na_percents = missing_cols(data_df)
    # return dependency table
    dependency_table = outdated_packages_list().loc[:,['name', 'version', 'latest_version']]    
    return jsonify(
        ingest_time = ex_times[0],        
        training_time = ex_times[1],
        na_percent = na_percents,
        dependencies = dependency_table.to_dict(orient='records'))

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)