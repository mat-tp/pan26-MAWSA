#!/usr/bin/env python
"""
PAN 2026 Multi-Author Writing Style Analysis - TIRA submission script.
"""

import argparse
import glob
import json
import gc
import os
import re
import sys

import warnings
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io import load_model, load_pipeline


def load_sentences_from_text(text: str) -> list:
    """Load sentences from TIRA input format (one sentence per line)."""
    return [line.strip() for line in text.strip().split("\n") if line.strip()]


def run_predict(input_dir: str, output_dir: str, model_path: str = None, pipeline_path: str = None):
    """Predict using fitted pipeline."""
    
        # ------------------------------------------------------------------
    # Resolve model / pipeline paths
    # ------------------------------------------------------------------

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
    
    # Verify feature count
    if hasattr(model, 'n_features_in_'):
        print(f"[TIRA] Model expects {model.n_features_in_} features")
        print(f"[TIRA] Pipeline provides {pipeline.n_features} features")
        
        assert pipeline.n_features == model.n_features_in_, \
            f"Feature mismatch! Pipeline has {pipeline.n_features}, " \
            f"model expects {model.n_features_in_}"
    
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
            # Extract features using FITTED pipeline
            sentence_features = pipeline.extract_batch(sentences)
            sentence_features_df = None
            if pd is not None:
                sentence_features_df = pd.DataFrame(
                    sentence_features,
                    columns=pipeline.feature_names,
                )

            # Create pairwise differences
            changes = []
            for i in range(len(sentences) - 1):
                if sentence_features_df is not None:
                    pair_vec = np.abs(
                        sentence_features_df.iloc[i] - sentence_features_df.iloc[i + 1]
                    )
                    X_pred = pair_vec.to_frame().T.astype(np.float32)
                else:
                    pair_vec = np.abs(sentence_features[i] - sentence_features[i + 1])
                    X_pred = pair_vec.reshape(1, -1).astype(np.float32)

                pred = model.predict(X_pred)[0]
                changes.append(int(pred))

        output_path = os.path.join(output_dir, f"solution-problem-{problem_id}.json")
        with open(output_path, "w") as f:
            json.dump({"changes": changes}, f)
        
        print(f"[TIRA] Problem {problem_id}: {len(sentences)} sentences → {len(changes)} predictions")


def main():
    parser = argparse.ArgumentParser(description="PAN 2026 TIRA Submission")
    parser.add_argument("-i", "--input", required=True, help="Input directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--model", default=None, help="Model path")
    parser.add_argument("--pipeline", default=None, help="Pipeline path")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[TIRA] Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    run_predict(args.input, args.output, args.model, args.pipeline)


if __name__ == "__main__":
    main()