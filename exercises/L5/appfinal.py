from flask import Flask, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/prediction')
def prediction():
    df = pd.read_csv('predictiondata.csv')
    with open('deployedmodel.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    y_pred = loaded_model.predict(df[['col1', 'col2']].values.reshape(-1,2))
    return str(y_pred)

app.run(host='0.0.0.0', port=8000)
