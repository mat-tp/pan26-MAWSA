"""
Main entry point for the Author Switch Detection system.
Modified to save fitted pipeline for TIRA deployment.
"""

import argparse
import gc
import os
import random
import sys
import time

import numpy as np
from scipy.sparse import hstack as sparse_hstack

# Add src/ to path so all package imports resolve from the correct root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import dataset_stats, flatten_problems, load_all, load_split
from evaluation.ablation import run_leave_one_out, run_single_group, save_ablation_csv
from evaluation.importance import logistic_coefficients, permutation_importance
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
from models.classifiers import MODEL_REGISTRY, train_model
from utils.config import (
    ABLATION_LOO_PATH,
    ABLATION_SGL_PATH,
    ACTIVE_FEATURE_GROUPS,
    CV_RESULTS_PATH,
    EVAL_RESULTS_PATH,
    IMPORTANCE_PATH,
    MIN_SENTENCES_PER_PROBLEM,
    MODEL_PATH,
    MODEL_SELECTION_PATH,
    MODELS_TO_COMPARE,
    PAIRWISE_MODE,
    PERM_IMP_REPEATS,
    PERM_IMP_TOP_K,
    ENABLE_PERM_IMPORTANCE,
    PLOT_TRAINING_LOG_PATH,
    PREDICTIONS_PATH,
    PRIMARY_MODEL,
    RAW_DIR,
    RANDOM_SEED,
    USE_PER_WORD_FW,
)
from utils.io import load_model, save_json, save_model, save_predictions, save_pipeline


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def _ram_mb() -> float:
    """Return current process RSS in MB (best-effort; NaN if psutil absent)."""
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
    """Delete references and trigger GC to release large arrays promptly."""
    for a in arrays:
        del a
    gc.collect()


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline() -> FeaturePipeline:
    """Instantiate the feature pipeline from config settings."""
    return FeaturePipeline(groups=ACTIVE_FEATURE_GROUPS, use_per_word_fw=USE_PER_WORD_FW)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_train_data(data_root: str, subset: float = 1.0) -> list:
    """
    Load training problems with optional subsampling.

    Keep subset ≤ 0.25 on 8 GB machines; raw text stays as Python strings
    until feature extraction, so most memory is in the feature matrix.
    """
    data     = load_all(data_root, splits=("train",))
    problems = flatten_problems(data)
    del data
    gc.collect()

    if subset < 1.0:
        random.seed(RANDOM_SEED)
        k        = max(1, int(len(problems) * subset))
        problems = random.sample(problems, k)
        problems = list(problems)
        gc.collect()

    print(f"\n[main] Total training problems (subset={subset}): {len(problems)}")
    for k, v in dataset_stats(problems).items():
        print(f"  {k}: {v}")

    _log_ram("after load_train_data")
    return problems


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_train(data_root: str, subset: float = 0.1):
    """
    Train all candidate models, select best by CV F1, and save best_model.pkl.
    Also saves the fitted feature pipeline for TIRA deployment.
    """
    t_start      = time.time()
    training_log = []

    def _checkpoint(step: str) -> None:
        training_log.append({
            "step":      step,
            "elapsed_s": round(time.time() - t_start, 1),
            "ram_mb":    _log_ram(step),
        })

    # ── Data loading and feature extraction ────────────────────────────────
    problems = load_train_data(data_root, subset=subset)
    _checkpoint("load_train_data")

    fp = build_pipeline()
    fp.describe()

    print("\n[main] Fitting pipeline on training sentences...")
    all_sentences = []
    for problem in problems:
        all_sentences.extend(problem["sentences"])
    
    fp.fit(all_sentences)
    print("[main] Pipeline fitted")

    print(f"\n[main] Building pairwise dataset (mode={PAIRWISE_MODE}) ...")
    ds = build_pairwise_dataset(
        problems, 
        feature_pipeline=fp,  # Make sure this is the exact fitted object
        min_sentences=MIN_SENTENCES_PER_PROBLEM,
        mode=PAIRWISE_MODE,
        use_cache=True,
    )
    _checkpoint("build_pairwise_dataset")

    # 🔥 SAVE THE FITTED PIPELINE (includes n-gram models, etc.)
    pipeline_path = os.path.join(os.path.dirname(MODEL_PATH), "feature_pipeline.pkl")
    save_pipeline(fp, pipeline_path)
    _checkpoint("save_pipeline")

    _free(problems)
    problems = None

    X_dense, y, problem_meta, groups = ds.to_memory()

    if ds.n_sparse > 0:
        print("[main] Loading sparse char-gram features into memory...")
        X_sparse = ds.X_sparse_unsafe()
        X = sparse_hstack([X_dense, X_sparse], format="csr")
    else:
        X = X_dense

    if X.dtype == np.float64:
        X = X.astype(np.float32)
        print("[main] Downcasted X to float32")
    _checkpoint("to_memory")

    # 🔥 VERIFY FEATURE COUNT
    print(f"[main] Feature matrix shape: {X.shape}")
    print(f"[main] Pipeline feature count: {fp.n_features}")
    assert X.shape[1] == fp.n_features, f"Feature mismatch: X has {X.shape[1]}, pipeline has {fp.n_features}"
    
    print(f"[main] X shape: {X.shape}  y: {(y == 0).sum()} same / {(y == 1).sum()} switch")

    try:
        plot_class_distribution(y)
    except Exception as e:
        print(f"[warning] Could not save class distribution plot: {e}")

    # ── 1. Cross-validation over all models ────────────────────────────────
    print(f"\n[main] Cross-validating {len(MODELS_TO_COMPARE)} model(s) ...")
    cv_results = compare_all_models(X, y, groups)
    save_json(cv_results, CV_RESULTS_PATH)
    _checkpoint("cross_validation")

    # ── 2. Train every model on the full dataset ───────────────────────────
    trained_models = {}
    valid_models   = {}  # models that completed CV with an F1 score

    print("\n[main] Training all candidate models on full dataset...")
    for model_name in MODELS_TO_COMPARE:
        if model_name not in MODEL_REGISTRY:
            print(f"[main] Skipping unavailable model {model_name}")
            continue
        try:
            print(f"   → Training {model_name}")
            model      = train_model(model_name, X, y)
            model_path = os.path.join(os.path.dirname(MODEL_PATH), f"{model_name}.pkl")
            save_model(model, model_path)
            trained_models[model_name] = model
            if model_name in cv_results and "f1" in cv_results[model_name]:
                valid_models[model_name] = cv_results[model_name]
        except Exception as e:
            print(f"[main] Failed to train/save {model_name}: {e}")
    _checkpoint("train_all_models")

    # ── 3. Select best model by CV F1 ─────────────────────────────────────
    if not valid_models:
        if PRIMARY_MODEL in MODEL_REGISTRY:
            best_model_name = PRIMARY_MODEL
            print(f"[main] No valid model found — falling back to {best_model_name}")
        else:
            available_models = [m for m in MODELS_TO_COMPARE if m in MODEL_REGISTRY]
            if not available_models:
                available_models = list(MODEL_REGISTRY)
            if not available_models:
                raise RuntimeError(
                    "No available classifier implementations found. "
                    "Install scikit-learn and/or the requested booster packages."
                )
            best_model_name = available_models[0]
            print(
                f"[main] No valid model found — primary model '{PRIMARY_MODEL}' is unavailable. "
                f"Falling back to '{best_model_name}'."
            )
        best_f1 = 0.0
        if best_model_name not in trained_models:
            try:
                trained_models[best_model_name] = train_model(best_model_name, X, y)
                save_model(
                    trained_models[best_model_name],
                    os.path.join(os.path.dirname(MODEL_PATH), f"{best_model_name}.pkl"),
                )
            except Exception as e:
                print(f"[main] CRITICAL: fallback model {best_model_name} also failed: {e}")
                raise
    else:
        best_model_name = max(valid_models, key=lambda n: valid_models[n]["f1"])
        best_f1         = valid_models[best_model_name]["f1"]
        print(f"\n[main] Best model: {best_model_name} (F1 = {best_f1:.4f})")

    best_model = trained_models[best_model_name]

    # ── 4. Save best model and selection metadata ──────────────────────────
    save_model(best_model, MODEL_PATH)
    _checkpoint("save_best_model")

    save_json({
        "best_model": best_model_name,
        "best_f1":    best_f1,
        "feature_count": int(X.shape[1]),
        "cv_results": cv_results,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }, MODEL_SELECTION_PATH)
    print(f"[main] Model selection saved to {MODEL_SELECTION_PATH}")

    if ENABLE_PERM_IMPORTANCE:
        # ── 5. Permutation importance ──────────────────────────────────────────
        print("\n[main] Computing permutation importance on best model...")
        MAX_IMP_ROWS = 5_000
        n_rows = X.shape[0]
        if n_rows > MAX_IMP_ROWS:
            rng            = np.random.default_rng(RANDOM_SEED)
            idx            = rng.choice(n_rows, MAX_IMP_ROWS, replace=False)
            X_imp, y_imp   = X[idx], y[idx]
        else:
            X_imp, y_imp = X, y

        imp = permutation_importance(
            best_model, X_imp, y_imp,
            feature_names=fp.feature_names,
            n_repeats=PERM_IMP_REPEATS,
            top_k=PERM_IMP_TOP_K,
        )
        save_json(imp, IMPORTANCE_PATH)
        _free(X_imp, y_imp)
        _checkpoint("permutation_importance")
    else:
        print("[main] Permutation importance skipped (ENABLE_PERM_IMPORTANCE=False)")

    if best_model_name == "logistic_regression":
        coef = logistic_coefficients(best_model, feature_names=fp.feature_names)
        save_json(coef, IMPORTANCE_PATH.replace(".json", "_coeff.json"))

    _checkpoint("done")
    plot_training_log(training_log, path=PLOT_TRAINING_LOG_PATH)

    return best_model, fp, X, y, problem_meta, groups, best_model_name


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

def run_ablation(data_root: str) -> None:
    """Run leave-one-out and single-group feature ablation studies."""
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


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def run_predict(data_root: str, model_path: str = None) -> list:
    """
    Load a trained model and generate predictions for a dataset.
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


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full(data_root: str, subset: float = 0.1) -> None:
    """Full pipeline: train (with model selection) → evaluate → predict."""
    best_model, fp, X, y, problem_meta, groups, best_model_name = run_train(data_root, subset=subset)
    _log_ram("run_full: after train")

    print("\n[main] Evaluating on training data (in-sample diagnostic) ...")
    eval_res = evaluate_model(best_model, X, y, model_name=best_model_name)

    print("\n[main] Per-difficulty breakdown:")
    pair_meta = expand_meta(problem_meta, y)
    diff_res  = evaluate_by_difficulty(best_model, X, y, pair_meta)
    eval_res["by_difficulty"] = diff_res

    print("\n[main] Error analysis:")
    errors = error_analysis(best_model, X, y, pair_meta)

    _free(X, y)
    X = y = None

    save_json({
        **eval_res,
        "errors_summary": {
            "false_positives": len(errors["false_positives"]),
            "false_negatives": len(errors["false_negatives"]),
        },
    }, EVAL_RESULTS_PATH)

    run_predict(data_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Author Switch Detector")
    parser.add_argument(
        "--mode",
        choices=["train", "ablation", "predict", "full"],
        default="full",
        help="Which pipeline to run",
    )
    parser.add_argument("--data",   default=RAW_DIR, help="Path to data root directory")
    parser.add_argument("--model",  default=None,    help="Path to saved model (predict mode only)")
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