"""
I/O helpers: save/load models, JSON, CSVs, and prediction files.
"""

import json
import os
import pickle

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    """Pickle a fitted model to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / 1024 ** 2
    print(f"[io] Model saved → {path}  ({size_mb:.1f} MB)")


def load_model(path: str):
    """Load a pickled model from *path*."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[io] Model file not found: {path}")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    print(f"[io] Model loaded ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# JSON
# ─────────────────────────────────────────────────────────────────────────────

def _json_default(obj):
    """Make numpy scalars and arrays JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def save_json(data, path: str) -> None:
    """Write *data* to a JSON file, handling numpy types automatically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_json_default)
    print(f"[io] JSON saved → {path}")


def load_json(path: str):
    """Load a JSON file and return the parsed object."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows, path: str) -> None:
    """
    Write a list of dicts to a CSV file.

    The fieldnames are inferred from the keys of the first row.
    """
    import csv

    if not rows:
        print(f"[io] save_csv: no rows to write ({path})")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[io] CSV saved → {path}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Predictions (JSONL — one JSON object per line, PAN-style)
# ─────────────────────────────────────────────────────────────────────────────

def save_predictions(predictions, path: str) -> None:
    """
    Write predictions to a JSONL file.

    Each element of *predictions* must be a dict with at least:
        {"problem_id": str, "changes": [int, ...]}
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred, default=_json_default) + "\n")
    print(f"[io] Predictions saved → {path}  ({len(predictions)} problems)")


def load_predictions(path: str):
    """Load a JSONL predictions file into a list of dicts."""
    predictions = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions
