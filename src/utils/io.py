"""
I/O utilities for saving/loading models, pipelines, and predictions.
"""

import json
import os
import pickle
import numpy as np


def save_model(model, path):
    """Save a trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f, protocol=4)
    print(f"[io] Model saved → {path}")


def load_model(path):
    """Load a trained model."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"[io] Model loaded ← {path}")
    return model


def save_pipeline(pipeline, path):
    """Save a fitted feature pipeline."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f, protocol=4)
    print(f"[io] Pipeline saved → {path}")


def load_pipeline(path):
    """Load a fitted feature pipeline."""
    with open(path, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"[io] Pipeline loaded ← {path}")
    return pipeline


def save_predictions(predictions, path):
    """Save predictions to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"[io] Predictions saved → {path}")


def load_predictions(path):
    """Load predictions from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    """Save any JSON-serializable data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[io] JSON saved → {path}")


def load_json(path):
    """Load JSON from path."""
    with open(path, 'r') as f:
        return json.load(f)