
from flask import Flask, request
import pandas as pd


app = Flask(__name__)


def readpandas(filename):
    thedata=pd.read_csv(filename)
    return thedata


@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello " + user + '\n'

@app.route('/size')
def size():
    filename = request.args.get('filename')
    df = readpandas(filename)
    return str(len(df.index))

@app.route('/summary')
def summary():
    filename = request.args.get('filename')
    df = readpandas(filename)
    return str(df.mean(axis=0))

app.run(host='0.0.0.0', port=8000)




