
# HW7: Airflow Pipelines — Training & Inference

This section contains instructions for running MLOps pipelines using **Apache Airflow** for both training and inference of the SentenceTransformers model on Ukrainian book data.

---

## ✅ Requirements

> It is recommended to use a virtual environment (venv or conda).

### 1. Install Apache Airflow (with constraints)

```bash
AIRFLOW_VERSION=2.8.1
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1,2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

---

### 2. Install other requirements

```bash
pip install -r requirements.txt
```

---

## 📁 DAGs Location

The Airflow DAGs are located in:

```
hw7/pipelines/airflow/
├── train_dag.py        # DAG for training the model
└── inference_dag.py    # DAG for inference and saving embeddings
```

You can symlink these DAGs into your Airflow DAG folder, for example:

```bash
ln -s $(pwd)/hw7/pipelines/airflow/train_dag.py ~/airflow/dags/
ln -s $(pwd)/hw7/pipelines/airflow/inference_dag.py ~/airflow/dags/
```

---

## ▶️ Running Airflow

1. Initialize the Airflow database:

```bash
airflow db init
```

2. Start the Airflow webserver:

```bash
airflow webserver --port 8080
```

3. In a separate terminal, start the scheduler:

```bash
airflow scheduler
```

4. Open Airflow UI at [http://localhost:8080](http://localhost:8080) and trigger the DAG manually.

---

## 📦 Outputs

- Trained model: `hw7/models/sentence_model.joblib`
- Inference results: `inference_results.csv`