import requests
data = {
    "instances": ["Що таке машинне навчання?"]
}
response = requests.post("http://localhost:8080/v1/models/tinyllama:predict", json=data)
print(response.json())
