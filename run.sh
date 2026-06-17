#!/usr/bin/env bash
set -e

# TIRA sets these environment variables automatically
# Use them to pass input/output directories to the Python script
exec python3 /app/src/tira_predict.py -i "$inputDataset" -o "$outputDir"