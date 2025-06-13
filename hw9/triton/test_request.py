import requests
import json

url = "http://localhost:8000/v2/models/tinyllama/infer"
headers = {"Content-Type": "application/json"}

data = {
    "inputs": [
        {
            "name": "text",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["Що таке машинне навчання?"]
        }
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
