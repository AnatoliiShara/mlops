from locust import HttpUser, task, between

PROMPTS = [
    "Порадь детектив на 5 книг",
    "Щось українською про саморозвиток",
    "Наукова фантастика у стилі Дюни",
]

class RestBookUser(HttpUser):
    wait_time = between(0.5, 1.5)          # think-time

    @task
    def ask_model(self):
        payload = {"prompt": PROMPTS[self.environment.runner.request_count % len(PROMPTS)]}
        self.client.post("/predict", json=payload, name="/predict")
