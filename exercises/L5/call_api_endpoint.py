import subprocess
import requests

# Call using curl
response1=subprocess.run(['curl', '127.0.0.1:8000?user=alex'],capture_output=True).stdout
print(response1)

response2=subprocess.run(['curl', '127.0.0.1:8000/size?filename=testdata.csv'],capture_output=True).stdout
print(response2)

response3=subprocess.run(['curl', '127.0.0.1:8000/summary?filename=testdata.csv'],capture_output=True).stdout
print(response3)

# Use requests
response4=requests.get('http://127.0.0.1:8000?user=Bradford').content
print(response4) 

response5=requests.get('http://127.0.0.1:8000/size?filename=testdata.csv').content
print(response5) 

response5=requests.get('http://127.0.0.1:8000/summary?filename=testdata.csv').content
print(response5) 

