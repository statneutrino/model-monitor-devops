from training import train_model
from scoring import score_model
from ingestion import merge_multiple_dataframe as ingest
from deployment import store_model_into_pickle as deploy
from reporting import score_model as cfm_report
from apicalls import run_api_call
import json
import os
import ast


with open('config.json','r') as f:
    config = json.load(f) 
deployment_dir = config['prod_deployment_path']
source_data_dir = config['input_folder_path']
features = ast.literal_eval(config['feature_names'])
output_folder_path = config['output_folder_path']
ingested_data_path = os.path.join(os.getcwd(), output_folder_path, 'finaldata.csv')
model_dir = os.path.join(os.getcwd(), config['output_model_path'])
clf_save_path = os.path.join(os.getcwd(), model_dir, 'trainedmodel.pkl')
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 

##################Check and read new data
def check_new_data(deployment_dir, source_data_dir):
    record_path = os.path.join(os.getcwd(), deployment_dir, 'ingestedfiles.txt')
    with open(record_path, 'r') as f:
        ingested_files = f.read().splitlines() 

    input_files = os.listdir(os.path.join(os.getcwd(), source_data_dir))

    new_files = set(input_files) - set(ingested_files)
    is_new_data = True if len(new_files) >= 1 else False
    return is_new_data, list(new_files)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def check_for_model_drift(deployment_dir, ingested_data_path, features):
    latest_score_path = os.path.join(os.getcwd(), deployment_dir, 'latestscore.txt')
    with open(latest_score_path, 'r') as f:
        current_score = float(f.read())

    # Evaluate F1 score based on new data
    new_score = score_model(deployment_dir, ingested_data_path, features, save = False)
    is_drift = True if new_score < current_score else False
    return is_drift, current_score, new_score

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here

##################Training and Re-deployment
#if you found evidence for model drift, score model on test data and save score, and then re-run the deployment.py script
# NOT APPROPRIATE TO TEST MODEL ON SAME DATA - ISSUE WITH INSTRUCTIONS - WHAT TO TEST DATA ON PREVENT training set leakage?

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

def run_full_pipeline(deployment_dir, source_data_dir, ingested_data_path, model_dir, clf_save_path, features):
    # Check for new data
    print("Checking for new data...")
    is_new_data, new_files = check_new_data(deployment_dir, source_data_dir)

    # If no new data - stop process
    if is_new_data is False:
        return "No new data found - process stopped"

    # Execute data ingestion
    print("New data found - running data ingestion")
    new_df = ingest(source_data_dir, output_folder_path)

    # Check for model drift - stop process if model drift not occurred
    print("Checking for model drift...")
    is_drift, current_score, new_score = check_for_model_drift(deployment_dir, ingested_data_path, features)
    if is_drift is False:
        return "No model drift - process stopped"
    
    # Retrain, save new score deploy model with ingestion and score records
    print("Model drift found - retraining ML model on new data...")
    train_model(ingested_data_path, clf_save_path)
    new_score = score_model(model_dir, ingested_data_path, features, save = True)
    print("New F1 score of {} is based on model trained data from {} folder".format(new_score, output_folder_path))
    deploy(model_dir, output_folder_path, deployment_dir)
    
    # Running diagnostics
    deployed_model_path = os.path.join(os.getcwd(), deployment_dir, 'trainedmodel.pkl')
    cfm_report(deployed_model_path, ingested_data_path, features)
    run_api_call(data_path = 'ingesteddata/finaldata.csv')
    return "Retraining, deployment and reporting completed"

if __name__ == '__main__':
    pass
    result = run_full_pipeline(deployment_dir, source_data_dir, ingested_data_path, model_dir, clf_save_path, features)
    print(result)
    





