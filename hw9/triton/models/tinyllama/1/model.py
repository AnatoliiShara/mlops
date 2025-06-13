import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32)
        self.model.eval()

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
            input_text = input_tensor.as_numpy()[0].decode("utf-8")

            inputs = self.tokenizer(input_text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)

            result_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            out_tensor = pb_utils.Tensor("output__0", np.array([result_text.encode("utf-8")], dtype=object))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses
