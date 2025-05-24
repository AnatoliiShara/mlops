FROM nvcr.io/nvidia/tritonserver:23.10-py3
RUN pip install --no-cache-dir sentencepiece sentence-transformers torch==2.0.1
