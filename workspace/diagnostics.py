
import pandas as pd
import timeit
import os
import ast
import pickle
import json
from statistics import mean, median, stdev
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 
target_name = config['target']
features = ast.literal_eval(config['feature_names'])

##################Function to get model predictions
def model_predictions(deployed_model_path, test_data_path, features, target_name = 'exited'):
    file_path = os.path.join(os.getcwd(), test_data_path)
    test_data = pd.read_csv(file_path)

    y = test_data[target_name].values.reshape(-1, 1).ravel()
    X = test_data.loc[:,features].values.reshape(-1, len(features))
    
    # Load trained model and calculate an F1 score with predictions
    with open(os.path.join(deployed_model_path), 'rb') as file:
        clf = pickle.load(file)
    y_pred = clf.predict(X)

    return y_pred

##################Function to get summary statistics
def dataframe_summary(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'] # reference for code is from Stackoverflow 
    df_numeric = df.select_dtypes(include=numerics)
    # "https://stackoverflow.com/questions/25039626/how-do-i-find-numeric-columns-in-pandas"
    means = list(df_numeric.apply(mean, axis=0))
    medians = list(df_numeric.apply(median, axis=0))
    stdevs = list(df_numeric.apply(stdev, axis=0))
    return means + medians + stdevs


##### Function for missing data summary
def missing_cols(df):
    na_counts = list(df.isna().sum())
    na_percents=[na_counts[i]/len(df.index) for i in range(len(na_counts))]
    return(na_percents)


##################Function to get timings
def execution_time():
    #calculate timing of ingestion.py
    starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingest_time = timeit.default_timer() - starttime

    #calculate timing of training.py
    starttime = timeit.default_timer()
    os.system('python3 training.py')
    training_time = timeit.default_timer() - starttime
    return [ingest_time, training_time]
    

##################Function to check dependencies
def outdated_packages_list():

    all_packages_json = subprocess.check_output(['pip', 'list', '--format', 'json'])
    all_packages = pd.read_json(all_packages_json)

    outdated_json = subprocess.check_output(['pip', 'list', '--outdated', '--format', 'json'])
    outdated_df = pd.read_json(outdated_json).loc[:,['name', 'version', 'latest_version']]

    uptodate_df = all_packages[~all_packages['name'].isin(outdated_df['name'])]
    uptodate_df.loc[:,'latest_version'] = uptodate_df['version']
    output_df = outdated_df.append(uptodate_df)

    return output_df.reset_index()
    

if __name__ == '__main__':
    print(model_predictions(deployed_model_path, test_data_path, features, target_name = 'exited'))
    df = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path))
    print(dataframe_summary(df))
    print(missing_cols(df))
    print(outdated_packages_list())





    
