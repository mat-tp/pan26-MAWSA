"""
I/O utilities: saving and loading models, JSON, and CSV results.

Keeping all I/O logic here makes it easy to swap serialisation formats
later without touching the rest of the codebase.
"""

import csv
import json
import os
import pickle


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path):
    """Pickle a fitted sklearn Pipeline to disk."""
    _makedirs(path)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[io] Model saved → {path}")


def load_model(path):
    """Load a pickled sklearn Pipeline from disk."""
    _check_exists(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def save_json(data, path, indent=2):
    """Serialise a dict/list to JSON."""
    _makedirs(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=_json_default)
    print(f"[io] JSON saved → {path}")


def load_json(path):
    """Load a JSON file and return the parsed object."""
    _check_exists(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def save_csv(rows, path, fieldnames=None):
    """
    Write a list of dicts to a CSV file.

    fieldnames defaults to the keys of the first row.
    """
    if not rows:
        print(f"[io] Nothing to write → {path}")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    _makedirs(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[io] CSV saved → {path}")


def load_csv(path):
    """Load a CSV file as a list of dicts."""
    _check_exists(path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# TIRA prediction output
# ---------------------------------------------------------------------------

def save_predictions(predictions, path):
    """
    Write predictions in PAN/TIRA format.

    predictions: list of dicts with keys: problem_id, changes (list of int)
    """
    _makedirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            record = {
                "id": pred["problem_id"],
                "changes": pred["changes"],
            }
            f.write(json.dumps(record) + "\n")
    print(f"[io] Predictions saved → {path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _makedirs(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _check_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[io] File not found: {path}")


def _json_default(obj):
    """Handle numpy types during JSON serialisation."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")
