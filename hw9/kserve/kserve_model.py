import kserve
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TinyLlamaModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.model.eval()
        self.ready = True
        return self
        
    def predict(self, request: Dict, headers: Dict = None) -> Dict:
        # Додано параметр headers, який очікує KServe
        instances = request["instances"]
        responses = []
        
        for instance in instances:
            prompt = instance["text"]
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=50)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)
            
        return {"predictions": responses}

if __name__ == "__main__":
    model = TinyLlamaModel("tinyllama")
    model.load()
    kserve.ModelServer().start([model])