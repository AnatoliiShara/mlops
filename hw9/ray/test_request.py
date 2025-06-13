# test_request_ray.py
import requests

data = {"prompt": "Що таке машинне навчання?"}
response = requests.post("http://localhost:8008/", json=data)
print(response.json())
