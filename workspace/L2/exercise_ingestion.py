import pandas as pd
from datetime import datetime
import os

directories=['/data1/','/data2/','/data3/']

final_dataframe = pd.DataFrame(columns=['col1', 'col2', 'col3'])

for directory in directories:
    filenames = os.listdir(os.getcwd()+directory)
    for each_filename in filenames:
        currentdf = pd.read_csv(os.getcwd()+directory+each_filename)
        final_dataframe=final_dataframe.append(currentdf).reset_index(drop=True)


result = final_dataframe.drop_duplicates()

print(result)