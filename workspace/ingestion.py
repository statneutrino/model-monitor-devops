import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import ast

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
column_names = ast.literal_eval(config['column_names'])

#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    input_dir = '/' + input_folder_path + '/'
    output_path = '/' + output_folder_path + '/'

    df_list = pd.DataFrame(columns=column_names)

    filenames = os.listdir(os.getcwd() + input_dir)

    file_dict = {}

    for each_filename in filenames:
        dateTimeObj = datetime.now()
        thetimenow = str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)
        temp_df = pd.read_csv(os.getcwd() + input_dir + each_filename)
        df_list=df_list.append(temp_df)
        file_dict[each_filename] = thetimenow
    
    merged_df = df_list.drop_duplicates()
    if not os.path.isdir(os.getcwd() + output_path):
        os.mkdir(os.getcwd() + output_path)
    merged_df.to_csv(os.getcwd() + output_path + 'finaldata.csv', index=False)
    
    with open(os.getcwd() + output_path + 'ingestedfiles.txt','w') as file:
        for item in file_dict.items():
            filename, _ = item # filename, date = item -- option to record dates
            file.write(str(filename) + '\n') # ': ' + date + '\n')

    return merged_df

if __name__ == '__main__':
    df = merge_multiple_dataframe(input_folder_path, output_folder_path)
    print(df)