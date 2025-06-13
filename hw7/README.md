
# HW7: Kubeflow Pipelines — Training & Inference

This project implements Kubeflow Pipelines for training and inference of a semantic embedding model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) using Ukrainian book descriptions.

## ✅ Requirements

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
hw7/
├── data/ukr_books_dataset.csv
├── models/                         # Trained models will be saved here
├── pipelines/
│   └── kubeflow/
│       ├── train_pipeline.py       # Kubeflow training pipeline
│       └── inference_pipeline.py   # Kubeflow inference pipeline
├── src/
│   ├── train.py
│   ├── infer.py
│   └── data_loader.py
├── train_pipeline.yaml             # compiled training pipeline
├── inference_pipeline.yaml         # compiled inference pipeline
├── compile_pipeline.py
└── compile_inference_pipeline.py
```

## ▶️ How to Compile Pipelines

**Compile training pipeline:**
```bash
python3 compile_pipeline.py
```

**Compile inference pipeline:**
```bash
python3 compile_inference_pipeline.py
```

## ☁️ Run on Kubeflow

1. Open the Kubeflow Pipelines dashboard (e.g., http://localhost:8080).
2. Go to **Pipelines → Upload Pipeline**
3. Upload `train_pipeline.yaml` → Run
4. Repeat with `inference_pipeline.yaml`

## 📈 Model Info

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Dataset: `ukr_books_dataset.csv`
- Output: CSV with sentence embeddings