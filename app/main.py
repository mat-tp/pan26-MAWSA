"""
Main entry point for the Author Switch Detection system.

Usage:
    # Train and evaluate (cross-validation + held-out)
    python main.py --mode train --data data/raw

    # Run ablation study
    python main.py --mode ablation --data data/raw

    # Predict on a new dataset (TIRA-style)
    python main.py --mode predict --data data/raw/test --model data/outputs/models/mlp.pkl

    # Full pipeline: train then predict
    python main.py --mode full --data data/raw

All outputs go to data/outputs/.
"""

import argparse
import os
import sys
import random

# Add app/ to path so imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.naive_bayes import GaussianNB

from data.loader import dataset_stats, flatten_problems, load_all, load_split
from evaluation.ablation import run_leave_one_out, run_single_group, save_ablation_csv
from evaluation.importance import logistic_coefficients, permutation_importance
from evaluation.metrics import (
    compare_all_models,
    evaluate_by_difficulty,
    evaluate_model,
    error_analysis,
)
from features.pipeline import FeaturePipeline
from features.pairwise import build_pairwise_dataset
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


def build_pipeline():
    """Create the feature pipeline from config settings."""
    return FeaturePipeline(
        groups=ACTIVE_FEATURE_GROUPS,
        use_per_word_fw=USE_PER_WORD_FW,
    )


def load_train_data(data_root, subset=1.0):
    """Load training problems with optional subsampling."""
    data = load_all(data_root, splits=("train",))
    problems = flatten_problems(data)

    # -----------------------------
    # SUBSET LOGIC
    # -----------------------------
    if subset < 1.0:
        random.seed(42)
        k = max(1, int(len(problems) * subset))
        problems = random.sample(problems, k)

    print(f"\n[main] Total training problems (subset={subset}): {len(problems)}")

    stats = dataset_stats(problems)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return problems


def run_train(data_root, subset=0.1):
    """Train the primary model and save to disk."""
    problems = load_train_data(data_root, subset=subset)
    fp = build_pipeline()
    fp.describe()

    print(f"\n[main] Building pairwise dataset (mode={PAIRWISE_MODE}) ...")
    X, y, meta, groups = build_pairwise_dataset(
        problems,
        fp,
        min_sentences=MIN_SENTENCES_PER_PROBLEM,
        mode=PAIRWISE_MODE,
    )
    print(
        f"[main] X shape: {X.shape}  y distribution: "
        f"{sum(y==0)} same / {sum(y==1)} switch"
    )

    # Cross-validate all models for comparison
    print(f"\n[main] Cross-validating {len(MODELS_TO_COMPARE)} model(s) ...")
    cv_results = compare_all_models(X, y, groups)
    save_json(cv_results, CV_RESULTS_PATH)

    # Train the final model on all training data
    print(f"\n[main] Training final model: {PRIMARY_MODEL}")
    model = train_model(PRIMARY_MODEL, X, y)
    save_model(model, MODEL_PATH)

    # Feature importance
    print(f"\n[main] Computing permutation importance ...")
    imp = permutation_importance(
        model,
        X,
        y,
        feature_names=fp.feature_names,
        n_repeats=PERM_IMP_REPEATS,
        top_k=PERM_IMP_TOP_K,
    )
    save_json(imp, IMPORTANCE_PATH)

    # Coefficient analysis if using LR
    if PRIMARY_MODEL == "logistic_regression":
        coef = logistic_coefficients(model, feature_names=fp.feature_names)
        save_json(coef, IMPORTANCE_PATH.replace(".json", "_coeff.json"))

    print(f"\n[main] Training complete. Model saved to {MODEL_PATH}")
    return model, fp, X, y, meta, groups


def run_ablation(data_root):
    """Run feature group ablation studies."""
    problems = load_train_data(data_root)

    print("\n[main] Running leave-one-out ablation ...")
    loo = run_leave_one_out(problems, model_name=PRIMARY_MODEL)
    save_ablation_csv(loo, ABLATION_LOO_PATH)

    print("\n[main] Running single-group ablation ...")
    sgl = run_single_group(problems, model_name=PRIMARY_MODEL)
    save_ablation_csv(sgl, ABLATION_SGL_PATH)

    print("\n[main] Ablation complete.")


def run_predict(data_root, model_path=None):
    """Load a trained model and generate predictions for a dataset."""
    model_path = model_path or MODEL_PATH
    model = load_model(model_path)
    fp = build_pipeline()

    data = load_all(data_root, splits=("test",))
    problems = flatten_problems(data)

    if not problems:
        # Fallback: treat data_root as a single split directory
        problems = load_split(data_root, difficulty="unknown")

    print(f"[main] Predicting on {len(problems)} problems ...")
    predictions = []

    for problem in problems:
        sentences = problem["sentences"]
        if len(sentences) < 2:
            predictions.append(
                {
                    "problem_id": problem["problem_id"],
                    "changes": [],
                }
            )
            continue

        vecs = fp.extract_document(sentences)
        changes = []
        for i in range(len(sentences) - 1):
            pair_vec = abs(vecs[i] - vecs[i + 1]).reshape(1, -1)
            pred = int(model.predict(pair_vec)[0])
            changes.append(pred)

        predictions.append(
            {
                "problem_id": problem["problem_id"],
                "changes": changes,
            }
        )

    save_predictions(predictions, PREDICTIONS_PATH)
    print(f"[main] Predictions written to {PREDICTIONS_PATH}")
    return predictions


def run_full(data_root, subset=0.1):
    """Full pipeline: train → evaluate → predict."""
    model, fp, X, y, meta, groups = run_train(data_root, subset=subset)

    print("\n[main] Evaluating on training data (in-sample diagnostic) ...")
    eval_res = evaluate_model(model, X, y, model_name=PRIMARY_MODEL)

    print("\n[main] Per-difficulty breakdown:")
    diff_res = evaluate_by_difficulty(model, X, y, meta)
    eval_res["by_difficulty"] = diff_res

    print("\n[main] Error analysis:")
    errors = error_analysis(model, X, y, meta)

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


def main():
    parser = argparse.ArgumentParser(description="Author Switch Detector")
    parser.add_argument(
        "--mode",
        choices=["train", "ablation", "predict", "full"],
        default="full",
        help="Which pipeline to run",
    )
    parser.add_argument("--data", default=RAW_DIR, help="Path to data root directory")
    parser.add_argument(
        "--model", default=None, help="Path to saved model (predict mode only)"
    )
    parser.add_argument(
        "--subset", type=float, default=0.2, help="Fraction of training data to use"
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
