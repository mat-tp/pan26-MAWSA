"""
Feature group ablation studies.

Two complementary experiments:

1. Leave-one-out ablation:
   Train with ALL feature groups minus one, measure F1 drop.
   A large drop when removing group X → group X is important.

2. Single-group evaluation:
   Train with only one feature group active, measure F1.
   Shows the standalone power of each group.

Both experiments use cross-validation to get reliable estimates.
Results are returned as dicts and saved as CSV for the paper.
"""

import csv
import os

import numpy as np

from app.evaluation.metrics import cross_validate_model
from app.features.pipeline import ALL_GROUPS, FeaturePipeline
from app.features.pairwise import build_pairwise_dataset
from app.models.classifiers import build_model


def run_leave_one_out(problems, model_name="mlp", n_splits=5, groups=None):
    """
    Leave-one-out ablation: remove one group at a time, report F1.

    Returns a list of dicts sorted by F1 drop (most important group first).
    """
    active_groups = groups or ALL_GROUPS

    # Baseline: all groups
    print("[ablation] Running baseline (all groups) ...")
    fp_full = FeaturePipeline(groups=active_groups)
    X_full, y_full, _, grp_full = build_pairwise_dataset(problems, fp_full)
    baseline = cross_validate_model(
        build_model(model_name), X_full, y_full, grp_full, n_splits
    )
    baseline_f1 = baseline["test_f1_mean"]
    print(f"  Baseline F1 = {baseline_f1:.4f}")

    results = [
        {
            "group": "ALL",
            "f1": baseline_f1,
            "f1_std": baseline["test_f1_std"],
            "f1_drop": 0.0,
        }
    ]

    for group in active_groups:
        ablated = [g for g in active_groups if g != group]
        print(f"[ablation] Without '{group}' ...")
        fp = FeaturePipeline(groups=ablated)
        X, y, _, grps = build_pairwise_dataset(problems, fp)
        res = cross_validate_model(build_model(model_name), X, y, grps, n_splits)
        drop = round(baseline_f1 - res["test_f1_mean"], 4)
        results.append(
            {
                "group": f"ALL - {group}",
                "f1": res["test_f1_mean"],
                "f1_std": res["test_f1_std"],
                "f1_drop": drop,
            }
        )
        print(f"  F1 = {res['test_f1_mean']:.4f}  drop = {drop:+.4f}")

    results.sort(key=lambda r: -r["f1_drop"])
    return results


def run_single_group(problems, model_name="mlp", n_splits=5, groups=None):
    """
    Single-group evaluation: train on each feature group in isolation.

    Returns a list of dicts sorted by F1 (best standalone group first).
    """
    active_groups = groups or ALL_GROUPS
    results = []

    for group in active_groups:
        print(f"[ablation] Only '{group}' ...")
        fp = FeaturePipeline(groups=[group])
        X, y, _, grps = build_pairwise_dataset(problems, fp)
        res = cross_validate_model(build_model(model_name), X, y, grps, n_splits)
        results.append(
            {
                "group": group,
                "f1": res["test_f1_mean"],
                "f1_std": res["test_f1_std"],
            }
        )
        print(f"  F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f}")

    results.sort(key=lambda r: -r["f1"])
    return results


def save_ablation_csv(results, path):
    """Write ablation results to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[ablation] Saved → {path}")
