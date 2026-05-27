#!/usr/bin/env python3
"""
TIRA prediction script for PAN 2026 Multi-Author Writing Style Analysis.
Called as: python predict_tira.py -i INPUT_DIR -o OUTPUT_DIR
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add app/ to path so we can import the project’s modules
APP_DIR = Path(__file__).resolve().parent / "software" / "app"
sys.path.insert(0, str(APP_DIR))

from data.loader import load_split  # you must ensure this works on plain text?
from features.pipeline import FeaturePipeline
from utils.io import load_model

# Paths baked into the image
MODEL_PATH = Path(__file__).resolve().parent / "software" / "model.pkl"
CONFIG_PATH = Path(__file__).resolve().parent / "software" / "feature_config.json"


def load_feature_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"groups": None, "use_per_word_fw": False}


def predict_document(model, fp, sentences):
    """Return a list of 0/1 for each consecutive sentence pair."""
    if len(sentences) < 2:
        return []
    vecs = fp.extract_document(sentences)
    changes = []
    for i in range(len(sentences) - 1):
        pair_vec = abs(vecs[i] - vecs[i + 1]).reshape(1, -1)
        pred = int(model.predict(pair_vec)[0])
        changes.append(pred)
    return changes


def process_level(model, fp, input_level_dir: Path, output_level_dir: Path):
    """Handle all problems in one difficulty level (easy/medium/hard)."""
    if not input_level_dir.exists():
        print(f"[tira] Warning: {input_level_dir} does not exist – skipping.")
        return

    output_level_dir.mkdir(parents=True, exist_ok=True)

    # Find all problem-*.txt files
    for txt_file in sorted(input_level_dir.glob("problem-*.txt")):
        problem_id = txt_file.stem.split("-")[1]  # e.g. "problem-12" -> "12"

        # Read sentences – adapt this to how your loader works.
        # If load_split() reads a directory, you might need to wrap the single file.
        # For now, assume you can read the file and split sentences yourself.
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        # Simple sentence splitting – replace with your own robust method
        # if your pipeline expects a list of sentence strings.
        sentences = [s.strip() for s in text.split("\n") if s.strip()]

        # Predict
        changes = predict_document(model, fp, sentences)

        # Write solution JSON
        solution = {"changes": changes}
        out_path = output_level_dir / f"solution-problem-{problem_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(solution, f)
        print(f"[tira] Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory (contains easy/, medium/, hard/)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory (will be created)"
    )
    args = parser.parse_args()

    input_base = Path(args.input)
    output_base = Path(args.output)

    # Load model and feature pipeline once
    print(f"[tira] Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    cfg = load_feature_config()
    fp = FeaturePipeline(
        groups=cfg.get("groups"),
        use_per_word_fw=cfg.get("use_per_word_fw", False),
    )
    print(f"[tira] Pipeline ready: {fp.n_features} features")

    for level in ["easy", "medium", "hard"]:
        process_level(
            model,
            fp,
            input_level_dir=input_base / level,
            output_level_dir=output_base / level,
        )

    print("[tira] Done.")


if __name__ == "__main__":
    main()
