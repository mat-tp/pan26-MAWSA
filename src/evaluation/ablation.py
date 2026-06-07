"""
Feature group ablation studies for author switch detection.

Provides leave-one-out and single-group ablation runs, each saving a
horizontal bar-chart PNG and a CSV of F1 scores to PLOTS_DIR / OUTPUTS_DIR.
"""

import csv
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics import cross_validate_model
from features.pipeline import ALL_GROUPS, FeaturePipeline
from features.pairwise import build_pairwise_dataset
from models.classifiers import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ablation] Plot saved → {path}")


def _get_style() -> str:
    available = plt.style.available
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"):
        if candidate in available:
            return candidate
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# Leave-one-out ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_leave_one_out(
    problems,
    model_name: str = "logistic_regression",
    n_splits: int = 5,
    groups=None,
    use_cache: bool = True,
):
    """
    Leave-one-out ablation: remove one feature group at a time, measure F1 drop.

    A large drop when removing group X → group X is important.

    Returns:
        list of dicts sorted by F1 drop (most important group first)
    """
    from utils.config import PLOT_ABLATION_LOO_PATH, PLOT_DPI, GENERATE_PLOTS

    active_groups = groups or ALL_GROUPS

    print("\n" + "=" * 70)
    print("Leave-One-Out Feature Group Ablation")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Groups: {active_groups}")
    print("=" * 70)

    # ── Baseline: all groups ───────────────────────────────────────────────
    print("\n[ablation] Running baseline (all groups) ...")
    fp_full = FeaturePipeline(groups=active_groups)
    ds_full = build_pairwise_dataset(problems, fp_full, use_cache=use_cache)
    X_full, y_full, _, grp_full = ds_full.to_memory()

    baseline = cross_validate_model(
        build_model(model_name), X_full, y_full, grp_full, n_splits
    )
    baseline_f1  = baseline["test_f1_mean"]
    baseline_std = baseline["test_f1_std"]
    print(f"  → Baseline F1 = {baseline_f1:.4f} ± {baseline_std:.4f}")

    results = [{
        "group":    "ALL",
        "f1":       baseline_f1,
        "f1_std":   baseline_std,
        "f1_drop":  0.0,
    }]

    # ── Remove each group ─────────────────────────────────────────────────
    for group in active_groups:
        ablated = [g for g in active_groups if g != group]
        print(f"\n[ablation] Removing '{group}' (using {ablated}) ...")

        fp  = FeaturePipeline(groups=ablated)
        ds  = build_pairwise_dataset(problems, fp, use_cache=use_cache)
        X, y, _, grps = ds.to_memory()

        res  = cross_validate_model(build_model(model_name), X, y, grps, n_splits)
        drop = round(baseline_f1 - res["test_f1_mean"], 4)

        results.append({
            "group":   f"ALL - {group}",
            "f1":      res["test_f1_mean"],
            "f1_std":  res["test_f1_std"],
            "f1_drop": drop,
        })
        print(f"  → F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f}  "
              f"(drop = {drop:+.4f})")

    # Sort by importance (largest drop first)
    results.sort(key=lambda r: -r["f1_drop"])

    print("\n" + "=" * 70)
    print("Ablation Results (sorted by importance)")
    print("=" * 70)
    print(f"  {'Feature Group':<30} {'F1':>8} {'Drop':>8}  Importance")
    print(f"  {'-' * 70}")
    for r in results[1:]:
        bar = "█" * max(0, int(r["f1_drop"] * 50))
        print(f"  {r['group']:<30} {r['f1']:>8.4f} {r['f1_drop']:>+8.4f}  {bar}")
    print("=" * 70)

    if GENERATE_PLOTS:
        _plot_ablation_loo(results, PLOT_ABLATION_LOO_PATH, PLOT_DPI)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-group ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_single_group(
    problems,
    model_name: str = "logistic_regression",
    n_splits: int = 5,
    groups=None,
    use_cache: bool = True,
):
    """
    Single-group evaluation: train on each feature group in isolation.

    Returns:
        list of dicts sorted by F1 (best standalone group first)
    """
    from utils.config import PLOT_ABLATION_SGL_PATH, PLOT_DPI, GENERATE_PLOTS

    active_groups = groups or ALL_GROUPS

    print("\n" + "=" * 70)
    print("Single-Group Feature Evaluation")
    print("=" * 70)
    print(f"Model: {model_name}")
    print("=" * 70)

    results = []

    for group in active_groups:
        print(f"\n[ablation] Testing '{group}' alone ...")

        fp  = FeaturePipeline(groups=[group])
        ds  = build_pairwise_dataset(problems, fp, use_cache=use_cache)
        X, y, _, grps = ds.to_memory()

        res        = cross_validate_model(build_model(model_name), X, y, grps, n_splits)
        n_features = X.shape[1]

        results.append({
            "group":      group,
            "n_features": n_features,
            "f1":         res["test_f1_mean"],
            "f1_std":     res["test_f1_std"],
        })
        print(f"  → F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f} "
              f"({n_features} features)")

    results.sort(key=lambda r: -r["f1"])

    print("\n" + "=" * 70)
    print("Single-Group Results (sorted by F1)")
    print("=" * 70)
    print(f"  {'Group':<20} {'Features':>10} {'F1':>10}  Performance")
    print(f"  {'-' * 70}")
    for r in results:
        bar = "█" * min(40, int(r["f1"] * 50))
        print(f"  {r['group']:<20} {r['n_features']:>10} {r['f1']:>10.4f}  {bar}")
    print("=" * 70)

    if GENERATE_PLOTS:
        _plot_ablation_single(results, PLOT_ABLATION_SGL_PATH, PLOT_DPI)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_ablation_loo(results: list, path: str, dpi: int = 150) -> None:
    """Horizontal bar chart of F1 drops (leave-one-out)."""
    # Skip the baseline row (drop == 0)
    rows = [r for r in results if r["group"] != "ALL"]
    if not rows:
        return

    labels = [r["group"].replace("ALL - ", "") for r in rows][::-1]
    drops  = [r["f1_drop"] for r in rows][::-1]
    stds   = [r["f1_std"]  for r in rows][::-1]

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.5)))
        colors = ["#D65F5F" if d > 0 else "#6ACC65" for d in drops]
        ax.barh(range(len(labels)), drops, xerr=stds,
                color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("F1 Drop (positive = important)")
        ax.set_title("Leave-One-Out Ablation — Feature Group Importance")
        ax.grid(True, axis="x", alpha=0.35)

    _save_fig(fig, path, dpi=dpi)


def _plot_ablation_single(results: list, path: str, dpi: int = 150) -> None:
    """Horizontal bar chart of standalone F1 per feature group."""
    if not results:
        return

    labels = [r["group"] for r in results][::-1]
    f1s    = [r["f1"]    for r in results][::-1]
    stds   = [r["f1_std"] for r in results][::-1]

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.5)))
        ax.barh(range(len(labels)), f1s, xerr=stds,
                color="#4878CF", alpha=0.85, capsize=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("F1 Score")
        ax.set_title("Single-Group Ablation — Standalone Predictive Power")
        ax.set_xlim(0, 1.0)
        ax.grid(True, axis="x", alpha=0.35)

    _save_fig(fig, path, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_ablation_csv(results: list, path: str) -> None:
    """Write ablation results to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not results:
        print("[ablation] No results to save")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"[ablation] Results saved → {path}")