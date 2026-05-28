FROM --platform=linux/amd64 python:3.11-slim

LABEL description="PAN 2026 MAWSA — Author Switch Detection"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# # NLTK resources
# RUN python -c "import nltk; \
#     nltk.download('punkt'); \
#     nltk.download('averaged_perceptron_tagger'); \
#     nltk.download('universal_tagset')"

# ---------------------------
# IMPORTANT: correct model path
# ---------------------------
COPY model.pkl /app/model.pkl
COPY feature_config.json /app/feature_config.json

# App code
COPY app/ /app/app/

# TIRA entrypoint
COPY tira/predict_tira.py /app/predict_tira.py
COPY tira/run.sh /run.sh
RUN chmod +x /run.sh

ENTRYPOINT ["/bin/bash", "/run.sh"]