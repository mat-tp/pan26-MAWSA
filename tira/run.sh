#!/bin/bash
set -e

python /app/predict_tira.py -i "${1:-/input}" -o "${2:-/output}"