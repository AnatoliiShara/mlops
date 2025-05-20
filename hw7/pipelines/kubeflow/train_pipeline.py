from kfp import dsl
from kfp.dsl import component, pipeline
import os

@component
def load_data_component(data_path: str) -> str:
    return data_path

@component
def train_component(data_path: str, model_dir: str):
    from src import train
    train.train_model(data_path, model_dir)

@dsl.pipeline(name="hw7-train-pipeline")
def training_pipeline(
    data_path: str = "data/ukr_books_dataset.csv",
    model_dir: str = "models/"
):
    data = load_data_component(data_path=data_path)
    train_component(data_path=data.output, model_dir=model_dir)

