import os, json, random
from locust import HttpUser, task, between

_PROMPTS = [
    "Порадьте детектив з неочікуваною розв’язкою",
    "Щось про розвиток лідерських навичок",
    "Психологія особистості українською",
]

class RecoUser(HttpUser):
    wait_time = between(0.2, 0.5)
    host = os.getenv("REST_HOST", "http://localhost:8000")

    @task
    def get_reco(self):
        payload = {"prompt": random.choice(_PROMPTS), "top_k": 5}
        with self.client.post("/recommend", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
