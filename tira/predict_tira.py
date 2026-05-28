#!/usr/bin/env python3

import argparse
import json
import sys
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


#  YOUR SENTENCE SPLITTER (fixed & clean)
def load_sentences(text):
    """Splits a problem text into individual sentences (line-based)."""
    sentences = []

    lines = text.strip().split("\n")

    for line in lines:
        clean_line = line.strip()
        if clean_line:
            sentences.append(clean_line)

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


def process_level(model, fp, input_level_dir: Path, output_level_dir: Path):
    if not input_level_dir.exists():
        print(f"[tira] Warning: {input_level_dir} missing")
        return

    output_level_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(input_level_dir.glob("problem-*.txt"))

    if not txt_files:
        print(f"[tira] Warning: No text files in {input_level_dir}")
        return

    for txt_file in txt_files:
        # FIXED ID EXTRACTION (robust)
        problem_id = txt_file.stem.replace("problem-", "")

        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = load_sentences(text)

        changes = predict_document(model, fp, sentences)

        output_file = output_level_dir / f"solution-problem-{problem_id}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"changes": changes}, f)

        print(f"[tira] Wrote {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    # ✅ TIRA SAFE MODE (IMPORTANT FIX)
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

    print("[tira] Pipeline ready")

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
