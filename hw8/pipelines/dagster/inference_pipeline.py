from dagster import asset, job, Definitions
from pathlib import Path
import os

@asset
def load_inference_data() -> str:
    data_path = "data/ukr_books_dataset.csv"
    assert Path(data_path).exists(), f"{data_path} not found"
    return data_path

@asset
def run_model_inference(load_inference_data: str) -> str:
    from src import infer as infer_module

    model_path = "models/sentence_model.joblib"
    assert Path(model_path).exists(), f"{model_path} not found"

    output_path = "inference_results.csv"
    infer_module.run_inference(load_inference_data, model_path, output_path)

    return output_path

@job
def inference_pipeline():
    run_model_inference(load_inference_data())

defs = Definitions(
    jobs=[inference_pipeline],
    assets=[load_inference_data, run_model_inference]
)
