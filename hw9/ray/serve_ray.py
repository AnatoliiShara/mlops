import ray
from ray import serve
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fastapi import Request
import uvicorn

@serve.deployment
class TinyLlamaRay:
    def __init__(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.model.eval()

    async def __call__(self, request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"response": result}

# === MAIN ===
if __name__ == "__main__":
    ray.init()
    serve.start(http_options={"host": "0.0.0.0", "port": 8008})

    # Вказуємо route_prefix тут, у serve.run
    serve.run(
        TinyLlamaRay.bind(),
        route_prefix="/",
    )
