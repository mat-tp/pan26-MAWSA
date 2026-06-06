"""
Main entry point for the Author Switch Detection system.

Usage:
    # Train and evaluate (cross-validation + held-out)
    python main.py --mode train --data data/raw

    # Run ablation study
    python main.py --mode ablation --data data/raw

    # Predict on a new dataset (TIRA-style)
    python main.py --mode predict --data data/raw/test --model data/outputs/models/model.pkl

    # Full pipeline: train then predict
    python main.py --mode full --data data/raw

All outputs go to data/outputs/.
All plots go to data/outputs/plots/.

Memory design
-------------
- build_pairwise_dataset() returns a PairwiseDataset (lazy memmap wrapper).
- ds.to_memory() materialises dense X into RAM; sparse stays on disk.
- X is downcast float64→float32 immediately to halve footprint.
- X is freed (del + gc.collect) before prediction pass.
- Training log dicts accumulate timing + RAM at each checkpoint, then a
  timeline PNG is saved at the end of run_train().
"""

import argparse
import gc
import os
import random
import sys
import time

import numpy as np

# Add src/ to path so all package imports resolve from the correct root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import (
    dataset_stats,
    flatten_problems,
    load_all,
    load_split,
)
from evaluation.ablation import (
    run_leave_one_out,
    run_single_group,
    save_ablation_csv,
)
from evaluation.importance import (
    logistic_coefficients,
    permutation_importance,
)
from evaluation.metrics import (
    compare_all_models,
    evaluate_by_difficulty,
    evaluate_model,
    error_analysis,
    plot_class_distribution,
    plot_training_log,
)
from features.pipeline import FeaturePipeline
from features.pairwise import build_pairwise_dataset, expand_meta
from models.classifiers import train_model
from utils.config import (
    ABLATION_LOO_PATH,
    ABLATION_SGL_PATH,
    ACTIVE_FEATURE_GROUPS,
    CV_RESULTS_PATH,
    EVAL_RESULTS_PATH,
    IMPORTANCE_PATH,
    MIN_SENTENCES_PER_PROBLEM,
    MODEL_PATH,
    MODELS_TO_COMPARE,
    PAIRWISE_MODE,
    PERM_IMP_REPEATS,
    PERM_IMP_TOP_K,
    PLOT_TRAINING_LOG_PATH,
    PREDICTIONS_PATH,
    PRIMARY_MODEL,
    RAW_DIR,
    RANDOM_SEED,
    USE_PER_WORD_FW,
)
from utils.io import (
    load_model,
    save_csv,
    save_json,
    save_model,
    save_predictions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    """Return current process RSS in MB (best-effort)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        return float("nan")


def _log_ram(tag: str) -> float:
    """Print and return current RAM usage in MB."""
    mb = _ram_mb()
    if not np.isnan(mb):
        print(f"[RAM] {tag}: {mb:.0f} MB")
    return mb


def _free(*arrays) -> None:
    """Delete references and run GC to release large arrays promptly."""
    for a in arrays:
        del a
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> FeaturePipeline:
    """Create the feature pipeline from config settings."""
    return FeaturePipeline(
        groups=ACTIVE_FEATURE_GROUPS,
        use_per_word_fw=USE_PER_WORD_FW,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_train_data(data_root: str, subset: float = 1.0) -> list:
    """
    Load training problems with optional subsampling.

    Memory note: problems is a list of dicts; the raw text stays as Python
    strings until feature extraction.  Keep subset ≤ 0.25 on 8 GB machines.
    """
    data     = load_all(data_root, splits=("train",))
    problems = flatten_problems(data)

    # Free the intermediate nested dict — we only need the flat list
    del data
    gc.collect()

    if subset < 1.0:
        random.seed(RANDOM_SEED)
        k        = max(1, int(len(problems) * subset))
        problems = random.sample(problems, k)
        problems = list(problems)   # compact the list so un-sampled dicts are GC'd
        gc.collect()

    print(f"\n[main] Total training problems (subset={subset}): {len(problems)}")
    stats = dataset_stats(problems)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    _log_ram("after load_train_data")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_train(data_root: str, subset: float = 0.1):
    """
    Train the primary model and save to disk.

    Key memory strategy
    -------------------
    1. build_pairwise_dataset() returns a PairwiseDataset (memmap-backed).
    2. ds.to_memory() loads dense X into RAM; X is downcast to float32.
    3. `problems` is freed immediately after pairwise construction.
    4. X/y are freed before the permutation importance sample is computed.
    5. A training_log list records elapsed time and RAM at each step;
       a PNG timeline is saved at the end.

    Returns:
        (model, fp, X, y, problem_meta, groups)
        X and y may be None if they were freed early — callers should check.
    """
    t_start = time.time()
    training_log: list = []

    def _checkpoint(step: str) -> None:
        training_log.append({
            "step":      step,
            "elapsed_s": round(time.time() - t_start, 1),
            "ram_mb":    _log_ram(step),
        })

    problems = load_train_data(data_root, subset=subset)
    _checkpoint("load_train_data")

    fp = build_pipeline()
    fp.describe()

    print(f"\n[main] Building pairwise dataset (mode={PAIRWISE_MODE}) ...")
    ds = build_pairwise_dataset(
        problems,
        fp,
        min_sentences=MIN_SENTENCES_PER_PROBLEM,
        mode=PAIRWISE_MODE,
        use_cache=True,
    )
    _checkpoint("build_pairwise_dataset")

    # Free raw problems — features are all we need now
    _free(problems)
    problems = None

    # Materialise dense matrix into RAM
    X, y, problem_meta, groups = ds.to_memory()

    # Downcast float64 → float32 to halve the feature-matrix footprint
    if X.dtype == np.float64:
        X = X.astype(np.float32)
        print("[main] Downcasted X to float32")
    _checkpoint("to_memory")

    print(
        f"[main] X shape: {X.shape}  y distribution: "
        f"{(y == 0).sum()} same / {(y == 1).sum()} switch"
    )

    # Save class distribution plot
    plot_class_distribution(y)

    # ── Cross-validate models one at a time ────────────────────────────────
    print(f"\n[main] Cross-validating {len(MODELS_TO_COMPARE)} model(s) ...")
    cv_results = compare_all_models(X, y, groups)
    save_json(cv_results, CV_RESULTS_PATH)
    _checkpoint("cross_validation")

    # ── Train final model ──────────────────────────────────────────────────
    print(f"\n[main] Training final model: {PRIMARY_MODEL}")
    model = train_model(PRIMARY_MODEL, X, y)
    save_model(model, MODEL_PATH)
    _checkpoint("train_final_model")

    # ── Permutation importance on a memory-budget sample ──────────────────
    print("\n[main] Computing permutation importance ...")
    MAX_IMP_ROWS = 5_000
    if len(X) > MAX_IMP_ROWS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X), MAX_IMP_ROWS, replace=False)
        X_imp, y_imp = X[idx], y[idx]
    else:
        X_imp, y_imp = X, y

    imp = permutation_importance(
        model,
        X_imp,
        y_imp,
        feature_names=fp.feature_names,
        n_repeats=PERM_IMP_REPEATS,
        top_k=PERM_IMP_TOP_K,
    )
    save_json(imp, IMPORTANCE_PATH)
    _free(X_imp, y_imp)
    _checkpoint("permutation_importance")

    if PRIMARY_MODEL == "logistic_regression":
        coef = logistic_coefficients(model, feature_names=fp.feature_names)
        save_json(coef, IMPORTANCE_PATH.replace(".json", "_coeff.json"))

    print(f"\n[main] Training complete. Model saved to {MODEL_PATH}")

    # ── Save training timeline PNG ─────────────────────────────────────────
    _checkpoint("done")
    plot_training_log(training_log, path=PLOT_TRAINING_LOG_PATH)

    return model, fp, X, y, problem_meta, groups


# ─────────────────────────────────────────────────────────────────────────────
# Ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(data_root: str) -> None:
    """
    Run feature group ablation studies.

    Each ablation run re-extracts features for a different feature subset via
    build_pairwise_dataset (which uses the disk cache, so the cost is mainly
    the to_memory() call, not re-extraction from text).
    """
    problems = load_train_data(data_root)

    print("\n[main] Running leave-one-out ablation ...")
    loo = run_leave_one_out(problems, model_name=PRIMARY_MODEL)
    save_ablation_csv(loo, ABLATION_LOO_PATH)
    gc.collect()

    print("\n[main] Running single-group ablation ...")
    sgl = run_single_group(problems, model_name=PRIMARY_MODEL)
    save_ablation_csv(sgl, ABLATION_SGL_PATH)
    gc.collect()

    print("\n[main] Ablation complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

def run_predict(data_root: str, model_path: str = None) -> list:
    """
    Load a trained model and generate predictions for a dataset.

    Memory note: predictions are accumulated as plain Python dicts (no large
    arrays), so memory stays flat regardless of corpus size.
    Only 2 adjacent sentence vectors live in RAM at any moment.
    """
    model_path = model_path or MODEL_PATH
    model      = load_model(model_path)
    fp         = build_pipeline()

    data     = load_all(data_root, splits=("validation",))
    problems = flatten_problems(data)
    del data
    gc.collect()

    if not problems:
        problems = load_split(data_root, difficulty="unknown")

    print(f"[main] Predicting on {len(problems)} problems ...")
    predictions = []

    for problem in problems:
        sentences = problem["sentences"]
        if len(sentences) < 2:
            predictions.append({"problem_id": problem["problem_id"], "changes": []})
            continue

        # Extract per-sentence vectors; only 2 adjacent rows live in RAM at once
        vecs    = fp.extract_batch(sentences)
        changes = []
        for i in range(len(sentences) - 1):
            pair_vec = np.abs(vecs[i] - vecs[i + 1]).reshape(1, -1).astype(np.float32)
            changes.append(int(model.predict(pair_vec)[0]))

        predictions.append({"problem_id": problem["problem_id"], "changes": changes})

        del vecs
        gc.collect()

    save_predictions(predictions, PREDICTIONS_PATH)
    print(f"[main] Predictions written to {PREDICTIONS_PATH}")
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_full(data_root: str, subset: float = 0.1) -> None:
    """Full pipeline: train → evaluate → predict."""
    model, fp, X, y, problem_meta, groups = run_train(data_root, subset=subset)
    _log_ram("run_full: after train")

    print("\n[main] Evaluating on training data (in-sample diagnostic) ...")
    eval_res = evaluate_model(model, X, y, model_name=PRIMARY_MODEL)

    print("\n[main] Per-difficulty breakdown:")
    # Expand compact problem_meta into per-pair meta for difficulty breakdown
    pair_meta = expand_meta(problem_meta, y)
    diff_res  = evaluate_by_difficulty(model, X, y, pair_meta)
    eval_res["by_difficulty"] = diff_res

    print("\n[main] Error analysis:")
    errors = error_analysis(model, X, y, pair_meta)

    # Free feature matrix before prediction pass
    _free(X, y)
    X = y = None

    save_json(
        {
            **eval_res,
            "errors_summary": {
                "false_positives": len(errors["false_positives"]),
                "false_negatives": len(errors["false_negatives"]),
            },
        },
        EVAL_RESULTS_PATH,
    )

    run_predict(data_root)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Author Switch Detector")
    parser.add_argument(
        "--mode",
        choices=["train", "ablation", "predict", "full"],
        default="full",
        help="Which pipeline to run",
    )
    parser.add_argument("--data",  default=RAW_DIR,  help="Path to data root directory")
    parser.add_argument("--model", default=None,      help="Path to saved model (predict mode only)")
    parser.add_argument(
        "--subset",
        type=float,
        default=0.05,
        help="Fraction of training data to use (0.35 recommended on 8 GB machines)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train(args.data, subset=args.subset)
    elif args.mode == "ablation":
        run_ablation(args.data)
    elif args.mode == "predict":
        run_predict(args.data, model_path=args.model)
    elif args.mode == "full":
        run_full(args.data, subset=args.subset)


if __name__ == "__main__":
    main()
