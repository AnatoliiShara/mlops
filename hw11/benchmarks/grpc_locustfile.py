from locust import User, task, between, events
import tritonclient.grpc as tc, numpy as np, random

_PROMPTS = [
    "Роман про психологію особистості",
    "Історичний пригодницький роман",
    "Щось схоже на Гаррі Поттера",
]
_SERVER = "localhost:8001"
_MODEL  = "tinyllama_ensemble"

class GrpcBookUser(User):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.cli = tc.InferenceServerClient(_SERVER)

    @task
    def ask(self):
        prompt = random.choice(_PROMPTS)
        inp = tc.InferInput("PROMPT", [1], "BYTES")
        inp.set_data_from_numpy(np.asarray([prompt.encode()], dtype=object))
        out = tc.InferRequestedOutput("FINAL", binary_data=False)
        _ = self.cli.infer(_MODEL, [inp], outputs=[out])
