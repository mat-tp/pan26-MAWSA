#!/usr/bin/env bash
set -euo pipefail

echo "[run.sh] All arguments: $@"
echo "[run.sh] Argument count: $#"

# Simply pass all arguments to Python
python /app/predict_tira.py "$@"
