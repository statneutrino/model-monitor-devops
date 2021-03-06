import requests
import json
import os


def run_api_call(data_path):
    with open('config.json','r') as f:
        config = json.load(f)

    output_model_dir = config['output_model_path']
    URL = config['url']

    # #Specify a URL that resolves to your workspace
    # URL = "http://127.0.0.1:8000/"

    # access_token = request.data.decode('UTF-8')

    #Call each API endpoint and store the responses
    response1 = requests.post(URL+'/prediction?inputdata=' + data_path).content
    response2 = requests.get(URL+'/scoring').content
    response3 = requests.get(URL+'/summarystats').content
    response4 = requests.get(URL+'/diagnostics').content


    #combine all API responses
    responses = [response1, response2, response3, response4]


    #write the responses to your workspace
    api_txt_path = os.path.join(os.getcwd(), output_model_dir, 'apireturns.txt')
    with open(api_txt_path, 'w') as f:
        for resp in responses:
            f.write(str(resp) + "\n")





