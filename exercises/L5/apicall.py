import subprocess
import requests

response1=subprocess.run(['curl', '127.0.0.1:8000/prediction'],capture_output=True).stdout
print(response1)

response2=requests.get('http://127.0.0.1:8000/prediction').content
print(response2) 
