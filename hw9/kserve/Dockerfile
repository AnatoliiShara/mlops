FROM continuumio/miniconda3:latest
RUN apt-get update && apt-get install -y git build-essential
WORKDIR /app
COPY kserve_model.py /app/kserve_model.py
RUN conda install -y python=3.10 pyyaml=5.4.1 && \
    pip install torch==2.0.1 transformers==4.36.2 sentencepiece && \
    pip install kserve==0.11.2
ENV MODEL_NAME=tinyllama
CMD ["python3", "kserve_model.py"]