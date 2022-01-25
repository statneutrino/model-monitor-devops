import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import ast
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
deployed_model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 
target_name = config['target']
features = ast.literal_eval(config['feature_names'])
output_model_path = os.path.join(config['output_model_path'])

##############Function for reporting
def score_model(deployed_model_path, test_data_path, features, target_name = 'exited'):
    # source for seaborn heatmap at Kaggle: https://www.kaggle.com/agungor2/various-confusion-matrix-plots
    # Load test_data to get y values
    test_data_path = os.path.join(os.getcwd(), test_data_path)
    test_data = pd.read_csv(test_data_path)
    y = test_data[target_name].values.reshape(-1, 1).ravel()
    
    #calculate a confusion matrix using the test data and the deployed model
    y_pred = model_predictions(deployed_model_path, test_data_path, features, target_name)

    #Generate the confusion matrix
    cf_matrix = confusion_matrix(y, y_pred)

    df_cfm = pd.DataFrame(cf_matrix, columns = np.unique(y), index = np.unique(y))
    df_cfm.index.name = 'Actual'
    df_cfm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    sns.set(font_scale=1.4) #for label size
    sns.heatmap(df_cfm, cmap="rocket_r", annot=True,annot_kws = {"size": 16} )

    plt.savefig(os.path.join(os.getcwd(), output_model_path,'confusionmatrix.png'))

    print(y)
    print(y_pred)
    return cf_matrix

if __name__ == '__main__':
    x = score_model(deployed_model_path, test_data_path, features)
    print(x)