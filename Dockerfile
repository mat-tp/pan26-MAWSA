FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-tira.txt .
RUN pip3 install --no-cache-dir -r requirements-tira.txt

# Install Python dependencies
COPY requirements-tira.txt .
RUN pip3 install --no-cache-dir -r requirements-tira.txt

# Copy pre-downloaded NLTK data (TIRA has no internet access)
COPY nltk_data /root/nltk_data

# Create output and model dirs
RUN mkdir -p /output /app/dataset/outputs/models /app/src

# Copy model artifacts
COPY dataset/outputs/models/best_model.pkl        /app/dataset/outputs/models/
COPY dataset/outputs/models/feature_pipeline.pkl  /app/dataset/outputs/models/
COPY dataset/outputs/models/variance_selector.pkl /app/dataset/outputs/models/

# Copy source code and entrypoint
COPY src/ /app/src/
COPY run.sh /app/run.sh
RUN sed -i 's/\r//' /app/run.sh && chmod +x /app/run.sh

ENTRYPOINT ["/app/run.sh"]