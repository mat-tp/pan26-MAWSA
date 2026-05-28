#!/usr/bin/env python3

import argparse
import json
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, "/app")

from app.features.pipeline import FeaturePipeline
from app.utils.io import load_model

MODEL_PATH = Path("/app/model.pkl")
CONFIG_PATH = Path("/app/feature_config.json")


def load_feature_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"groups": None, "use_per_word_fw": False}


def load_sentences(text: str):
    sentences = []
    for line in text.strip().split("\n"):
        clean = line.strip()
        if clean:
            sentences.append(clean)
    return sentences


def predict_document(model, fp, sentences):
    if len(sentences) < 2:
        return []
    vecs = fp.extract_document(sentences)
    changes = []
    for i in range(len(sentences) - 1):
        pair_vec = abs(vecs[i] - vecs[i + 1]).reshape(1, -1)
        pred = int(model.predict(pair_vec)[0])
        changes.append(pred)
    return changes


def process_files(model, fp, txt_files, output_dir: Path):
    """Process a list of .txt files and write output into output_dir (flat)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for txt_file in sorted(txt_files):
        match = re.search(r"problem-(\d+)", txt_file.stem)
        problem_id = match.group(1) if match else txt_file.stem

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = load_sentences(text)
        changes = predict_document(model, fp, sentences)

        output_file = output_dir / f"solution-problem-{problem_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"changes": changes}, f)
        print(f"[tira] Wrote {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    input_base = Path(
        args.input if args.input else os.environ.get("inputDataset", "/input")
    )
    output_base = Path(
        args.output if args.output else os.environ.get("outputDir", "/output")
    )

    print(f"[tira] input: {input_base}")
    print(f"[tira] output: {output_base}")

    model = load_model(MODEL_PATH)
    cfg = load_feature_config()
    fp = FeaturePipeline(
        groups=cfg.get("groups"),
        use_per_word_fw=cfg.get("use_per_word_fw", False),
    )
    print(f"[tira] Pipeline ready: {fp.n_features} features")

    LEVELS = ["easy", "medium", "hard"]

    # Check whether the input uses difficulty subdirectories
    has_subdirs = any((input_base / lvl).is_dir() for lvl in LEVELS)

    if has_subdirs:
        # --- Subdirectory layout (local test / some TIRA datasets) ---
        for level in LEVELS:
            level_dir = input_base / level
            if not level_dir.exists():
                print(f"[tira] Warning: {level_dir} missing, skipping")
                continue
            txt_files = list(level_dir.glob("*.txt"))
            if not txt_files:
                print(f"[tira] Warning: No text files in {level_dir}")
                continue
            # Write into output_base/level/ AND output_base/ (flat)
            # so TIRA finds the files regardless of which it checks
            process_files(model, fp, txt_files, output_base / level)
            process_files(model, fp, txt_files, output_base)
    else:
        # --- Flat layout (TIRA smoketest) ---
        txt_files = list(input_base.glob("*.txt"))
        if not txt_files:
            # Recurse one level down just in case
            txt_files = list(input_base.glob("**/*.txt"))
        if not txt_files:
            print(f"[tira] Warning: No text files found in {input_base}")
        else:
            process_files(model, fp, txt_files, output_base)

    print("[tira] Done.")


if __name__ == "__main__":
    main()
