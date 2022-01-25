import training
import scoring
from ingestion import merge_multiple_dataframe as ingest
import deployment
import diagnostics
import reporting
import json
import os
import ast


with open('config.json','r') as f:
    config = json.load(f) 
deployment_dir = config['prod_deployment_path']
source_data_dir = config['input_folder_path']
features = ast.literal_eval(config['feature_names'])
output_folder_path = config['output_folder_path']

##################Check and read new data
def check_new_data(deployment_dir, source_data_dir):
    record_path = os.path.join(os.getcwd(), deployment_dir, 'ingestedfiles.txt')
    with open(record_path, 'r') as f:
        ingested_files = f.read().splitlines() 

    input_files = os.listdir(os.path.join(os.getcwd(), source_data_dir))

    new_files = set(input_files) - set(ingested_files)
    is_new_data = True if len(new_files) >= 1 else False
    return (is_new_data, list(new_files))

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
latest_score_path = os.path.join(os.getcwd(), deployment_dir, 'latestscore.txt')
with open(latest_score_path, 'r') as f:
    current_score = float(f.read())

# Evaluate F1 score based on new data
new_score = scoring.score_model(deployment_dir, test_data_path, features)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here



##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

def run_full_pipeline(deployment_dir, source_data_dir):
    is_new_data, new_files = check_new_data(deployment_dir, source_data_dir)
    if is_new_data is False:
        return "No new data"
    new_df = ingest(source_data_dir, output_folder_path)
    return new_df



if __name__ == '__main__':
    print(run_full_pipeline(deployment_dir, source_data_dir))





