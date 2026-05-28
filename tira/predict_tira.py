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
LEVELS = ["easy", "medium", "hard"]


def load_feature_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"groups": None, "use_per_word_fw": False}


def load_sentences(text: str):
    return [l.strip() for l in text.strip().split("\n") if l.strip()]


def predict_document(model, fp, sentences):
    if len(sentences) < 2:
        return []
    vecs = fp.extract_document(sentences)
    changes = []
    for i in range(len(sentences) - 1):
        pair_vec = abs(vecs[i] - vecs[i + 1]).reshape(1, -1)
        changes.append(int(model.predict(pair_vec)[0]))
    return changes


def process_files(model, fp, txt_files, output_dir: Path):
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


def scan_inputs(input_base: Path):
    """
    Returns list of (txt_file, level_or_None) tuples.
    Tries subdir layout first; falls back to flat root scan.
    Also prints a full directory tree for debugging.
    """
    print(f"[tira] Scanning input tree:")
    for root, dirs, files in os.walk(str(input_base)):
        rel = Path(root).relative_to(input_base)
        print(f"[tira]   {rel}/  files={files}")

    # Try subdirs first — only count a level as "present" if it has files
    results = {}  # level -> [Path]
    for level in LEVELS:
        level_dir = input_base / level
        if not level_dir.exists():
            continue
        # Accept .txt or bare problem-N files
        files = (
            list(level_dir.glob("*.txt"))
            or list(level_dir.glob("*.text"))
            or [
                f
                for f in level_dir.iterdir()
                if f.is_file() and re.search(r"problem-\d+", f.name)
            ]
        )
        if files:
            results[level] = files

    if results:
        return results  # {level: [files]}

    # Flat fallback — search root and one level deep
    flat = (
        list(input_base.glob("*.txt"))
        or list(input_base.glob("*.text"))
        or list(input_base.glob("**/*.txt"))
    )
    if flat:
        print(f"[tira] Using flat layout: {len(flat)} file(s) in root")
        return {None: flat}  # None signals flat output

    print(f"[tira] Warning: No input files found anywhere under {input_base}")
    return {}


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

    print(f"[tira] input:  {input_base}")
    print(f"[tira] output: {output_base}")

    model = load_model(MODEL_PATH)
    cfg = load_feature_config()
    fp = FeaturePipeline(
        groups=cfg.get("groups"),
        use_per_word_fw=cfg.get("use_per_word_fw", False),
    )
    print(f"[tira] Pipeline ready: {fp.n_features} features")

    found = scan_inputs(input_base)

    for level, files in found.items():
        if level is None:
            # flat layout — write directly into output_base
            process_files(model, fp, files, output_base)
        else:
            # subdir layout — write into subdir AND flat root
            process_files(model, fp, files, output_base / level)
            process_files(model, fp, files, output_base)  # TIRA validates flat root

    print("[tira] Done.")


if __name__ == "__main__":
    main()
