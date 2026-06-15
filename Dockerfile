# Use official Python runtime with system dependencies built-in
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install minimal compilers required for Python C-extensions and file sanitization
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download required NLTK datasets into the image layer
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('averaged_perceptron_tagger_eng'); \
    nltk.download('universal_tagset')"

# Create required external integration and output paths
RUN mkdir -p /output /app/dataset/outputs/models /app/src

# Copy prediction script environment and source code
COPY run.sh /app/run.sh
COPY src/ /app/src/

# Cross-platform sanity fix: Clean hidden Windows line-endings and make executable
RUN sed -i 's/$//' /app/run.sh && chmod +x /app/run.sh

# Defines the static execution entry point for production TIRA submissions
ENTRYPOINT ["/app/run.sh"]

# Default fallback flag when no platform arguments are supplied
CMD ["--help"]
