#!/usr/bin/env bash
set -euo pipefail
python3 /app/src/tira_predict.py -i "$inputDataset" -o "$outputDir"