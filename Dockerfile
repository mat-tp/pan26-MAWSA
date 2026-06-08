FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NLTK resources required by POS features
RUN python - <<EOF
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("universal_tagset")
EOF

# Source code
COPY src/ ./src/

# Run script
COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

# Model directory
RUN mkdir -p /app/dataset/outputs/models

# Trained artifacts
COPY dataset/outputs/models/best_model.pkl \
    /app/dataset/outputs/models/best_model.pkl

COPY dataset/outputs/models/feature_pipeline.pkl \
    /app/dataset/outputs/models/feature_pipeline.pkl

# TIRA output directory
RUN mkdir -p /output

ENTRYPOINT ["/app/run.sh"]
CMD ["-h"]