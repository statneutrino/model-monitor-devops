from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    number = request.args.get('number')
    return str(int(number)+1)+'\n'

app.run(host='0.0.0.0', port=8000)