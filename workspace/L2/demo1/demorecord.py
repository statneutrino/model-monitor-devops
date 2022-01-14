import pandas as pd
from datetime import datetime

sourcelocation='./recorddatasource/'
filename='recordkeepingdemo.csv'
outputlocation='records.txt'

data=pd.read_csv(sourcelocation+filename)

dateTimeObj=datetime.now()
thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

allrecords=[sourcelocation,filename,len(data.index),thetimenow]

MyFile=open(outputlocation,'w')
for element in allrecords:
     MyFile.write(str(element) + '\n')