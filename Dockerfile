FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies with PINNED VERSIONS
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy trained artifacts (must exist from training)
COPY dataset/outputs/models/best_model.pkl /app/data/outputs/models/best_model.pkl
COPY dataset/outputs/models/feature_pipeline.pkl /app/data/outputs/models/feature_pipeline.pkl

# Create output directory (TIRA will mount this)
RUN mkdir -p /output

# Entrypoint for TIRA
ENTRYPOINT ["python", "/app/src/tira_predict.py"]
CMD ["-h"]

# FROM python:3.11-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PYTHONPATH=/app/src

# WORKDIR /app

# # System dependencies (keep minimal but ML-safe)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     gcc \
#     git \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies first (cache layer)
# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# # Copy project
# COPY . .

# # Ensure NLTK data is available
# RUN python -c "import nltk; \
#     nltk.download('punkt'); \
#     nltk.download('stopwords'); \
#     nltk.download('averaged_perceptron_tagger')"

# # Default working command
# CMD ["python", "-m", "src.main", "--help"]