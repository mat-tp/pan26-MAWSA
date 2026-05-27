"""
Loads PAN 2026 MAWSA problems from disk.

Each problem is a plain-text file (one sentence per line) paired with a
truth JSON file containing a "changes" list of 0/1 values.

Expected directory layout:
    <root>/<difficulty>/<split>/problem-<id>.txt
    <root>/<difficulty>/<split>/truth-problem-<id>.json
"""

import json
import os
import re

# Matches filenames like "problem-0042.txt"
PROBLEM_FILE = re.compile(r"^problem-(\d+)\.txt$")


def load_sentences(text):
    """ Splits a problem text into individual sentences.
    """
    sentences = []

    lines = text.strip().split("\n")  # split the full text into lines

    for line in lines:
        clean_line = line.strip()  # remove leading/trailing whitespace
        if clean_line:  # only add non-empty lines
            sentences.append(clean_line)

    return sentences


def load_problem(txt_path, truth_path, difficulty):
    """
    Load one problem: read sentences from txt_path and labels from truth_path.

    Returns a dict with keys:
        problem_id, difficulty, sentences, changes
    Returns None if the file is missing, malformed, or fails the sanity check.
    """
    if not os.path.exists(truth_path):
        print(f"[loader] Missing truth file: {truth_path} — skipping.")
        return None

    try:
        with open(txt_path, encoding="utf-8") as f:
            sentences = load_sentences(f.read())

        with open(truth_path, encoding="utf-8") as f:
            changes = [int(c) for c in json.load(f)["changes"]]

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[loader] Error reading {txt_path}: {exc} — skipping.")
        return None

    # There must be exactly (n_sentences - 1) change labels.
    if len(sentences) - 1 != len(changes):
        print(
            f"[loader] Skipping {os.path.basename(txt_path)}: "
            f"{len(sentences)} sentences but {len(changes)} labels."
        )
        return None

    problem_id = re.search(r"problem-(\d+)\.txt$", txt_path).group(1)

    return {
        "problem_id": problem_id,
        "difficulty": difficulty,
        "sentences": sentences,
        "changes": changes,
    }


def load_split(path, difficulty):
    """Load all problems from a single directory (one difficulty × split)."""
    problems = []

    if not os.path.isdir(path):
        print(f"[loader] Directory not found: {path}")
        return problems

    for fname in sorted(os.listdir(path)):
        match = PROBLEM_FILE.match(fname)
        if not match:
            continue

        pid = match.group(1)
        txt_path = os.path.join(path, fname)
        truth_path = os.path.join(path, f"truth-problem-{pid}.json")

        problem = load_problem(txt_path, truth_path, difficulty)
        if problem:
            problems.append(problem)

    return problems


def load_all(root, difficulties=("easy", "medium", "hard"), splits=("train",)):
    """
    Load every difficulty × split combination under root.

    Returns a dict keyed by "<difficulty>_<split>", e.g. "easy_train".
    """
    data = {}
    for diff in difficulties:
        for split in splits:
            key = f"{diff}_{split}"
            path = os.path.join(root, diff, split)
            data[key] = load_split(path, diff)
            print(f"[loader] {key}: {len(data[key])} problems loaded.")
    return data


def flatten_problems(data_dict):
    """Merge all problems from a load_all() result into a single list."""
    all_problems = []
    for problems in data_dict.values():
        all_problems.extend(problems)
    return all_problems


def dataset_stats(problems):
    """Return a summary dict describing a list of problems."""
    if not problems:
        return {}

    n_probs = len(problems)
    n_sents = sum(len(p["sentences"]) for p in problems)
    n_pairs = sum(len(p["changes"]) for p in problems)
    n_switches = sum(sum(p["changes"]) for p in problems)
    word_lens = [len(s.split()) for p in problems for s in p["sentences"]]
    avg_words = round(sum(word_lens) / len(word_lens), 1) if word_lens else 0.0

    return {
        "n_problems": n_probs,
        "n_sentences": n_sents,
        "n_pairs": n_pairs,
        "n_switches": n_switches,
        "switch_rate_pct": round(n_switches / n_pairs * 100, 2) if n_pairs else 0.0,
        "avg_sents_per_problem": round(n_sents / n_probs, 1),
        "avg_words_per_sent": avg_words,
    }
