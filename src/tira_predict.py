#!/usr/bin/env python
"""
PAN 2026 Multi-Author Writing Style Analysis - TIRA submission script.
"""

import argparse
import glob
import json
import os
import re
import sys
import warnings

import numpy as np

# Add the src directory to path so pickle can find 'models' module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from utils.io import load_model, load_pipeline


def get_selector_path(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), "variance_selector.pkl")


def load_sentences_from_text(text: str) -> list:
    """Load sentences from TIRA input format (one sentence per line)."""
    return [line.strip() for line in text.strip().split("\n") if line.strip()]


def run_predict(input_dir: str, output_dir: str, model_path: str = None, pipeline_path: str = None):
    """Predict using fitted pipeline."""
    
    
    # Resolve model / pipeline paths

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    default_model_dir = os.path.join(
        repo_root,
        "dataset",
        "outputs",
        "models"
    )

    if model_path is None:
        model_path = os.path.join(
            default_model_dir,
            "best_model.pkl"
        )

    if pipeline_path is None:
        pipeline_path = os.path.join(
            default_model_dir,
            "feature_pipeline.pkl"
        )

    print(f"[TIRA] Loading model:\n       {model_path}")
    print(f"[TIRA] Loading pipeline:\n       {pipeline_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[TIRA] Error: Model not found: {model_path}"
        )

    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"[TIRA] Error: Pipeline not found: {pipeline_path}"
        )

    
    print(f"[TIRA] Loading model from: {model_path}")
    model = load_model(model_path)
    
    print(f"[TIRA] Loading fitted pipeline from: {pipeline_path}")
    pipeline = load_pipeline(pipeline_path)

    selector_path = get_selector_path(model_path)
    if not os.path.exists(selector_path):
        raise FileNotFoundError(
            f"[TIRA] Variance selector not found: {selector_path}. "
            "Train a model first so feature selection is saved alongside the model."
        )
    print(f"[TIRA] Loading variance selector from: {selector_path}")
    selector = load_model(selector_path)

    if hasattr(model, "n_features_in_"):
        expected = int(model.n_features_in_)
        sample_shape = None
        sample_text = ["This is a sample sentence.", "Another quick test sentence."]
        sample_feats = pipeline.extract_batch(sample_text)
        sample_feats = selector.transform(sample_feats)
        sample_shape = sample_feats.shape[1]
        if sample_shape != expected:
            raise ValueError(
                f"Feature mismatch before TIRA prediction: model expects {expected} features, "
                f"but pipeline+selector produces {sample_shape}."
            )

    os.makedirs(output_dir, exist_ok=True)
    
    # Find all problem files
    txt_files = sorted(glob.glob(os.path.join(input_dir, "problem-*.txt")))
    
    if not txt_files:
        print(f"[TIRA] Warning: No problem-*.txt files found in {input_dir}")
        return
    
    print(f"[TIRA] Found {len(txt_files)} problem files")
    
    for txt_file in txt_files:
        basename = os.path.basename(txt_file)
        match = re.search(r"problem-(\d+)\.txt", basename)
        if not match:
            continue
        problem_id = match.group(1)
        
        with open(txt_file, "r", encoding="utf-8") as f:
            sentences = load_sentences_from_text(f.read())
        
        if len(sentences) < 2:
            changes = []
        else:
            # Extract features using fitted pipeline
            sentence_features = pipeline.extract_batch(sentences)
            sentence_features = selector.transform(sentence_features)

            # Create pairwise differences
            changes = []
            for i in range(len(sentences) - 1):
                X_pred = np.abs(sentence_features[i] - sentence_features[i + 1])
                X_pred = X_pred.reshape(1, -1).astype(np.float32)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="X does not have valid feature names",
                        category=UserWarning,
                    )
                    pred = model.predict(X_pred)[0]
                changes.append(int(pred))

        output_path = os.path.join(output_dir, f"solution-problem-{problem_id}.json")
        with open(output_path, "w") as f:
            json.dump({"changes": changes}, f)
        
        print(f"[TIRA] Problem {problem_id}: {len(sentences)} sentences → {len(changes)} predictions")


def main():
    parser = argparse.ArgumentParser(description="PAN 2026 TIRA Submission")
    parser.add_argument("-i", "--input", 
                       nargs='?',
                       default=None,
                       help="Input directory")
    parser.add_argument("-o", "--output", 
                       nargs='?',
                       default=None,
                       help="Output directory")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--pipeline", default=None, help="Pipeline path")
    
    # Use parse_known_args to handle any unexpected arguments
    args, unknown = parser.parse_known_args()
    
    # Determine input directory with multiple fallbacks
    input_dir = args.input
    if not input_dir:
        input_dir = os.environ.get("inputDataset")
    if not input_dir:
        input_dir = os.environ.get("TIRA_INPUT_DIRECTORY")
    if not input_dir:
        input_dir = "/tmp/input"
    
    # Determine output directory with multiple fallbacks
    output_dir = args.output
    if not output_dir:
        output_dir = os.environ.get("outputDir")
    if not output_dir:
        output_dir = os.environ.get("TIRA_OUTPUT_DIRECTORY")
    if not output_dir:
        output_dir = "/tmp/output"
    
    print(f"[TIRA] Input directory: {input_dir}")
    print(f"[TIRA] Output directory: {output_dir}")
    print(f"[TIRA] Environment inputDataset: {os.environ.get('inputDataset', 'NOT SET')}")
    print(f"[TIRA] Environment outputDir: {os.environ.get('outputDir', 'NOT SET')}")
    
    if not os.path.exists(input_dir):
        print(f"[TIRA] Warning: Input directory not found: {input_dir}")
        print(f"[TIRA] Creating input directory...")
        os.makedirs(input_dir, exist_ok=True)
    
    run_predict(input_dir, output_dir, args.model, args.pipeline)

if __name__ == "__main__":
    main()