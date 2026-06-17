#!/usr/bin/env bash
set -e
echo "run.sh executed with args: $@"
echo "inputDataset=$inputDataset"
echo "outputDir=$outputDir"
exec python3 /app/src/tira_predict.py "$@"