from kfp import dsl
from kfp.dsl import component, pipeline

@component
def load_data_for_inference(data_path: str) -> str:
    return data_path

@component
def inference_component(data_path: str, model_path: str, output_path: str):
    from src import infer
    infer.run_inference(data_path, model_path, output_path)

@pipeline(name="hw7-inference-pipeline")
def inference_pipeline(
    data_path: str = "data/ukr_books_dataset.csv",
    model_path: str = "models/sentence_model.joblib",
    output_path: str = "inference_results.csv"
):
    data = load_data_for_inference(data_path=data_path)
    inference_component(
        data_path=data.output,
        model_path=model_path,
        output_path=output_path
    )
