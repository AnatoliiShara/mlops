FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y git build-essential
WORKDIR /app
COPY model/tinyllama.py /app/tinyllama.py

#  ─── deps ─────────────────────────────────────────────────────────
RUN conda install -y python=3.10 pyyaml=5.4.1 && \
    pip install torch==2.0.1 \
               sentencepiece \
               transformers==4.36.2 \
               seldon-core==1.15.0

ENV SELDON_API_TYPE=REST
CMD ["seldon-core-microservice", "tinyllama.TinyLlama"]
