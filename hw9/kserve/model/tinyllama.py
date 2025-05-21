from kserve import Model, model_server
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

class TinyLlama(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.ready = False

    def load(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.model.eval()
        self.ready = True

    def predict(self, request: dict) -> dict:
        prompt = request["instances"][0]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"predictions": [response]}

if __name__ == "__main__":
    model = TinyLlama("tinyllama")
    model.load()
    model_server.start([model])
