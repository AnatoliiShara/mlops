import requests
import json

prompt = "Give me a book recommendation about machine learning."
response = requests.post(
    "http://localhost:8000/api/v1.0/predictions",
    data=json.dumps({"data": {"ndarray": [[prompt]]}}),
    headers={"Content-Type": "application/json"},
)

print("Response:", response.json())
