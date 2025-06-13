import numpy as np, torch, triton_python_backend_utils as pb_utils
from sentence_transformers import AutoTokenizer, AutoModelForCausalLM

class TritonPythonModel:
    def initialize(self, args):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        ).eval()

    def execute(self, requests):
        # ---- зберемо batch текстів ----
        batch_txt = []
        for req in requests:
            tsr = pb_utils.get_input_tensor_by_name(req, "input__0")
            batch_txt.append(tsr.as_numpy().astype(str)[0])

        # ---- forward одним батчем ----
        tok = self.tokenizer(batch_txt, return_tensors="pt", padding=True)
        with torch.no_grad():
            out_ids = self.model.generate(**tok, max_new_tokens=50)
        decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)

        # ---- розіб’ємо відповіді назад по запитах ----
        responses = []
        for txt in decoded:
            tensor = pb_utils.Tensor("output__0",
                                     np.array([txt.encode("utf-8")], dtype=object))
            responses.append(pb_utils.InferenceResponse([tensor]))
        return responses
