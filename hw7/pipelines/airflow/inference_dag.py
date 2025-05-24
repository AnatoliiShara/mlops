import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from airflow import DAG
from airflow.decorators import task
from datetime import datetime

with DAG(
    dag_id="hw7_infer_sentence_model",
    start_date=datetime(2025, 5, 20),
    schedule=None,
    catchup=False,
    description="Run inference with SentenceTransformer model on book descriptions"
) as dag:

    @task
    def infer():
        from hw7.src import infer as infer_module

        data_path = "hw7/data/ukr_books_dataset.csv"
        model_path = "hw7/models/sentence_model.joblib"
        output_path = "hw7/inference_results.csv"

        infer_module.run_inference(data_path, model_path, output_path)

    infer()
