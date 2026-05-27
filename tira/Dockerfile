FROM --platform=linux/amd64 python:3.11-slim

LABEL description="PAN 2026 MAWSA — Author Switch Detection (stylometric, CPU-only)"
LABEL maintainer="matlala-tp@redblinkinglight"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True); nltk.download('universal_tagset', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy TIRA scripts
COPY tira/predict_tira.py .
COPY tira/run.sh /run.sh
RUN chmod +x /run.sh

# Create the software/ directory and copy your application code
RUN mkdir -p /app/software/app
COPY app/ /app/software/app/

# Copy your trained model and feature config (adjust paths as needed)
# If you have them in the project root, use:
COPY model.pkl /app/software/model.pkl
COPY feature_config.json /app/software/feature_config.json

# If the files are elsewhere, change the source accordingly.
# If you don't have feature_config.json, comment out that line (the script handles missing file).

ENTRYPOINT ["/run.sh"]