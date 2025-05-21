from transformers import LlamaTokenizer, LlamaForCausalLM      # ← явно
from seldon_core.user_model import SeldonComponent
import torch

class TinyLlama(SeldonComponent):
    def __init__(self):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32
        )
        self.model.eval()

    def predict(self, X, features_names=None):
        prompt = X[0][0]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)
        return [[self.tokenizer.decode(output[0], skip_special_tokens=True)]]
