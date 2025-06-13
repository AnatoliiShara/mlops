from dagster import asset, job, Definitions
from pathlib import Path
import os

@asset
def load_training_data() -> str:
    data_path = "data/ukr_books_dataset.csv"
    assert Path(data_path).exists(), f"{data_path} not found"
    return data_path

@asset
def train_model(load_training_data: str) -> str:
    from src import train as train_module

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    train_module.train_model(load_training_data, model_dir)

    model_path = os.path.join(model_dir, "sentence_model.joblib")
    return model_path


@job
def training_pipeline():
    model_path = train_model(load_training_data())

defs = Definitions(
    jobs=[training_pipeline],
    assets=[load_training_data, train_model]
)
