FROM --platform=linux/amd64 python:3.11-slim

LABEL description="PAN 2026 MAWSA — Author Switch Detection (stylometric, CPU-only)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); \
    nltk.download('universal_tagset', quiet=True); nltk.download('punkt_tab', quiet=True)"

# App code
COPY app/ /app/software/app/

# Model + feature config
COPY model.pkl         /app/software/model.pkl
COPY feature_config.json /app/software/feature_config.json

# TIRA entrypoint — pull from tira/ subfolder
COPY tira/predict_tira.py /app/predict_tira.py
COPY tira/run.sh          /run.sh
RUN chmod +x /run.sh

ENTRYPOINT ["/bin/bash", "/run.sh"]