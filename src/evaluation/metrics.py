"""
Cross-validation, evaluation metrics, and diagnostic plots for author switch detection.

All plots are saved to PLOTS_DIR (never plt.show() — safe for headless/Docker runs).
Matplotlib backend is forced to Agg at import time.
"""

import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")   # headless — must come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate


# ─────────────────────────────────────────────────────────────────────────────
# Internal plot helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """Save *fig* to *path*, creating parent dirs as needed, then close."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[metrics] Plot saved → {path}")


def _get_style() -> str:
    """Return a valid matplotlib style name."""
    try:
        from utils.config import PLOT_STYLE
        style = PLOT_STYLE
    except ImportError:
        style = "seaborn-v0_8-whitegrid"
    # Fall back gracefully if the style isn't available
    available = plt.style.available
    if style in available:
        return style
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"):
        if candidate in available:
            return candidate
    return "default"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    model,
    X,
    y,
    groups,
    n_splits: int = 5,
    scoring=None,
    return_models: bool = False,
    threshold: float = 0.5,
):
    """
    Perform group-aware cross-validation to prevent data leakage.

    Groups = problem IDs ensure all pairs from the same document stay together.

    Returns:
        dict with mean/std for each metric
    """
    if scoring is None:
        scoring = {
            "accuracy":          "accuracy",
            "precision":         "precision",
            "recall":            "recall",
            "f1":                "f1",
            "roc_auc":           "roc_auc",
            "average_precision": "average_precision",
        }

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            groups=groups,
            scoring=scoring,
            return_estimator=return_models,
            n_jobs=-1,
            error_score="raise",
        )

    results = {}
    for key in scoring:
        scores = cv_results[f"test_{key}"]
        results[f"test_{key}_mean"] = float(np.mean(scores))
        results[f"test_{key}_std"]  = float(np.std(scores))
        results[f"test_{key}_all"]  = scores.tolist()

    if "train_f1" in cv_results:
        results["train_f1_mean"] = float(np.mean(cv_results["train_f1"]))
        results["train_f1_std"]  = float(np.std(cv_results["train_f1"]))

    if return_models:
        results["models"] = cv_results["estimator"]

    print(f"\n[metrics] {n_splits}-fold CV results (group-aware):")
    print(f"  F1:          {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}")
    print(f"  ROC-AUC:     {results['test_roc_auc_mean']:.4f} ± {results['test_roc_auc_std']:.4f}")
    print(f"  Avg Prec:    {results['test_average_precision_mean']:.4f} ± {results['test_average_precision_std']:.4f}")
    print(f"  Accuracy:    {results['test_accuracy_mean']:.4f} ± {results['test_accuracy_std']:.4f}")
    print(f"  Precision:   {results['test_precision_mean']:.4f} ± {results['test_precision_std']:.4f}")
    print(f"  Recall:      {results['test_recall_mean']:.4f} ± {results['test_recall_std']:.4f}")

    if "train_f1_mean" in results:
        gap = results["train_f1_mean"] - results["test_f1_mean"]
        if gap > 0.05:
            print(f"  ⚠ Overfitting detected: train-test F1 gap = {gap:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Single-model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str = "model", threshold: float = 0.5):
    """
    Comprehensive evaluation on a test set.  Saves plots to PLOTS_DIR.

    Returns:
        dict of metrics
    """
    from utils.config import PLOTS_DIR, PLOT_DPI, GENERATE_PLOTS

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_classification_report(y_test, y_pred, y_prob, metrics)

    if GENERATE_PLOTS:
        with plt.style.context(_get_style()):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(f"Evaluation — {model_name}", fontsize=13, fontweight="bold")

            plot_confusion_matrix(y_test, y_pred, ax=axes[0])
            plot_roc_curve(y_test, y_prob, metrics.get("roc_auc"), ax=axes[1])
            plot_precision_recall_curve(y_test, y_prob, metrics.get("avg_precision"), ax=axes[2])

        _save_fig(
            fig,
            os.path.join(PLOTS_DIR, f"eval_{model_name}.png"),
            dpi=PLOT_DPI,
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive classification metrics."""
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"]       = float(roc_auc_score(y_true, y_prob))
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pass  # only one class present

    metrics["n_samples"]    = int(len(y_true))
    metrics["n_positive"]   = int(np.sum(y_true))
    metrics["n_negative"]   = int(np.sum(y_true == 0))
    metrics["positive_rate"] = float(np.mean(y_true))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
    })

    return metrics


def print_classification_report(y_true, y_pred, y_prob=None, metrics=None):
    """Print formatted classification results."""
    if metrics is None:
        metrics = compute_metrics(y_true, y_pred, y_prob)

    print("\n" + "=" * 60)
    print("Classification Results")
    print("=" * 60)
    print(f"  Samples:      {metrics['n_samples']}")
    print(f"  Positives:    {metrics['n_positive']} ({metrics['positive_rate']:.1%})")
    print(f"  Negatives:    {metrics['n_negative']} ({1 - metrics['positive_rate']:.1%})")
    print("-" * 60)
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    if "avg_precision" in metrics:
        print(f"  Avg Precision:{metrics['avg_precision']:.4f}")
    print("-" * 60)
    print("  Confusion Matrix:")
    print(f"    TN={metrics['true_negatives']:5d}  FP={metrics['false_positives']:5d}")
    print(f"    FN={metrics['false_negatives']:5d}  TP={metrics['true_positives']:5d}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(model, X_val, y_val, metric: str = "f1"):
    """
    Find optimal classification threshold and save the sweep plot.

    Returns:
        (optimal_threshold, best_metric_value)
    """
    from utils.config import PLOT_THRESHOLD_PATH, PLOT_DPI, GENERATE_PLOTS

    if not hasattr(model, "predict_proba"):
        print("[metrics] Model doesn't support predict_proba. Using default threshold 0.5")
        return 0.5, None

    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    _FN = {"f1": f1_score, "precision": precision_score, "recall": recall_score}
    fn = _FN.get(metric, f1_score)
    scores = [fn(y_val, (y_prob >= t).astype(int), zero_division=0) for t in thresholds]

    best_idx       = int(np.argmax(scores))
    best_threshold = float(thresholds[best_idx])
    best_score     = float(scores[best_idx])

    if GENERATE_PLOTS:
        with plt.style.context(_get_style()):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(thresholds, scores, "b-", linewidth=2, label=metric.upper())
            ax.axvline(best_threshold, color="r", linestyle="--",
                       label=f"Best {best_threshold:.3f} ({metric}={best_score:.3f})")
            ax.axvline(0.5, color="gray", linestyle=":", label="Default (0.5)")
            ax.set_xlabel("Threshold")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Threshold Optimisation — {metric.upper()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        _save_fig(fig, PLOT_THRESHOLD_PATH, dpi=PLOT_DPI)

    print(f"\n[metrics] Optimal threshold: {best_threshold:.3f} ({metric}={best_score:.4f})")
    return best_threshold, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot helpers (axes-based — callers manage the figure)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, ax=None, normalize: bool = True):
    """Draw confusion matrix on *ax* (creates a figure if None)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix" + (" (normalised)" if normalize else ""))
    plt.colorbar(im, ax=ax)

    classes    = ["Same Author", "Switch"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    fmt    = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    return ax


def plot_roc_curve(y_true, y_prob, auc_score=None, ax=None):
    """Draw ROC curve on *ax*."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    label = f"ROC (AUC = {auc_score:.3f})" if auc_score is not None else "ROC"
    ax.plot(fpr, tpr, "b-", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_precision_recall_curve(y_true, y_prob, avg_precision=None, ax=None):
    """Draw precision-recall curve on *ax*."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    label = f"PR (AP = {avg_precision:.3f})" if avg_precision is not None else "PR"
    ax.plot(recall, precision, "b-", linewidth=2, label=label)

    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="r", linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# Standalone plot: save confusion matrix / ROC / PR as individual PNGs
# ─────────────────────────────────────────────────────────────────────────────

def save_diagnostic_plots(y_true, y_pred, y_prob, prefix: str = "eval"):
    """
    Save confusion matrix, ROC curve, and PR curve as separate PNG files.
    prefix is prepended to each filename.
    """
    from utils.config import PLOTS_DIR, PLOT_DPI

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_confusion_matrix(y_true, y_pred, ax=ax)
    _save_fig(fig, os.path.join(PLOTS_DIR, f"{prefix}_confusion.png"), dpi=PLOT_DPI)

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(6, 5))
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
        plot_roc_curve(y_true, y_prob, auc, ax=ax)
    _save_fig(fig, os.path.join(PLOTS_DIR, f"{prefix}_roc.png"), dpi=PLOT_DPI)

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(6, 5))
        ap = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
        plot_precision_recall_curve(y_true, y_prob, ap, ax=ax)
    _save_fig(fig, os.path.join(PLOTS_DIR, f"{prefix}_pr.png"), dpi=PLOT_DPI)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_models(X, y, groups, n_splits: int = 5):
    """
    Cross-validate every model in MODELS_TO_COMPARE.

    Saves a bar-chart comparison plot (PLOT_CV_PATH) after all runs.

    Returns:
        dict mapping model_name → CV result dict
    """
    from utils.config import MODELS_TO_COMPARE, PLOT_CV_PATH, PLOT_DPI, GENERATE_PLOTS
    from models.classifiers import build_model

    results = {}
    print(f"\n{'=' * 60}")
    print("Model Comparison (cross-validation)")
    print(f"{'=' * 60}")

    for name in MODELS_TO_COMPARE:
        print(f"\n[metrics] Evaluating: {name}")
        try:
            model = build_model(name)
            cv    = cross_validate_model(model, X, y, groups, n_splits=n_splits)
            results[name] = cv
        except Exception as exc:
            print(f"[metrics] {name} failed: {exc}")
            results[name] = {"error": str(exc)}

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{'Model':<25} {'F1':>8} {'±':>6} {'AUC':>8}")
    print(f"{'-' * 60}")
    for name, res in results.items():
        if "error" not in res:
            print(f"  {name:<23} {res['test_f1_mean']:>8.4f} "
                  f"{res['test_f1_std']:>6.4f} {res['test_roc_auc_mean']:>8.4f}")
        else:
            print(f"  {name:<23} ERROR: {res['error'][:30]}")
    print(f"{'=' * 60}\n")

    # ── Bar chart ──────────────────────────────────────────────────────────
    if GENERATE_PLOTS:
        _plot_cv_comparison(results, PLOT_CV_PATH, PLOT_DPI)

    return results


def _plot_cv_comparison(results: dict, path: str, dpi: int = 150) -> None:
    """Bar chart of F1 and AUC across models."""
    valid = {k: v for k, v in results.items() if "error" not in v}
    if not valid:
        return

    names  = list(valid.keys())
    f1s    = [valid[n]["test_f1_mean"]      for n in names]
    f1_std = [valid[n]["test_f1_std"]       for n in names]
    aucs   = [valid[n]["test_roc_auc_mean"] for n in names]

    x    = np.arange(len(names))
    width = 0.35

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.8), 5))
        bars1 = ax.bar(x - width / 2, f1s,  width, yerr=f1_std, capsize=4,
                       label="F1", color="#4878CF", alpha=0.85)
        bars2 = ax.bar(x + width / 2, aucs, width,
                       label="ROC-AUC", color="#6ACC65", alpha=0.85)

        ax.set_ylabel("Score")
        ax.set_title("Cross-Validation Model Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.35)

        # Annotate bars
        for bar in bars1:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    _save_fig(fig, path, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Training log plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_log(log: list, path: str = None, dpi: int = 150) -> None:
    """
    Save a timeline plot from a list of training-step dicts.

    Each dict should contain at least {"step": str, "elapsed_s": float}.
    Optional keys: "ram_mb", "n_pairs", "f1_mean".

    Called by main.run_train() after the training loop.
    """
    from utils.config import PLOT_TRAINING_LOG_PATH, PLOT_DPI, GENERATE_PLOTS

    if not GENERATE_PLOTS or not log:
        return

    path = path or PLOT_TRAINING_LOG_PATH
    dpi  = dpi  or PLOT_DPI

    steps   = [e.get("step", f"step_{i}") for i, e in enumerate(log)]
    elapsed = [e.get("elapsed_s", 0.0)    for e in log]
    ram     = [e.get("ram_mb", None)      for e in log]
    has_ram = any(r is not None for r in ram)

    n_axes = 2 if has_ram else 1
    with plt.style.context(_get_style()):
        fig, axes = plt.subplots(1, n_axes, figsize=(6 * n_axes, 4))
        if n_axes == 1:
            axes = [axes]

        # Elapsed timeline
        ax = axes[0]
        ax.barh(range(len(steps)), elapsed, color="#4878CF", alpha=0.8)
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(steps, fontsize=8)
        ax.set_xlabel("Elapsed time (s)")
        ax.set_title("Training Timeline")
        ax.grid(True, axis="x", alpha=0.35)

        # RAM usage
        if has_ram:
            ax2 = axes[1]
            ram_vals = [r if r is not None else 0.0 for r in ram]
            ax2.plot(range(len(steps)), ram_vals, "o-", color="#D65F5F", linewidth=2)
            ax2.set_xticks(range(len(steps)))
            ax2.set_xticklabels(steps, rotation=35, ha="right", fontsize=8)
            ax2.set_ylabel("RAM (MB)")
            ax2.set_title("Memory Usage")
            ax2.grid(True, alpha=0.35)

        fig.suptitle("Training Log", fontsize=13, fontweight="bold")

    _save_fig(fig, path, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Class distribution plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y, meta=None, path: str = None, dpi: int = 150) -> None:
    """
    Save a bar chart of class distribution, optionally split by difficulty.
    """
    from utils.config import PLOT_CLASS_DIST_PATH, PLOT_DPI, GENERATE_PLOTS

    if not GENERATE_PLOTS:
        return

    path = path or PLOT_CLASS_DIST_PATH
    dpi  = dpi  or PLOT_DPI

    with plt.style.context(_get_style()):
        if meta is None or not any("difficulty" in m for m in meta):
            # Simple overall distribution
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]
            ax.bar(["Same Author", "Switch"], counts, color=["#4878CF", "#D65F5F"], alpha=0.85)
            for i, c in enumerate(counts):
                ax.text(i, c + max(counts) * 0.01, str(c), ha="center", va="bottom")
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution")
            ax.grid(True, axis="y", alpha=0.35)
        else:
            # Split by difficulty
            difficulties = sorted({m.get("difficulty", "unknown") for m in meta})
            y_arr = np.array(y)
            same   = []
            switch = []
            for d in difficulties:
                idx = [i for i, m in enumerate(meta) if m.get("difficulty") == d]
                same.append(int(np.sum(y_arr[idx] == 0)))
                switch.append(int(np.sum(y_arr[idx] == 1)))

            x     = np.arange(len(difficulties))
            width = 0.35
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - width / 2, same,   width, label="Same Author", color="#4878CF", alpha=0.85)
            ax.bar(x + width / 2, switch, width, label="Switch",      color="#D65F5F", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(difficulties)
            ax.set_ylabel("Count")
            ax.set_title("Class Distribution by Difficulty")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.35)

    _save_fig(fig, path, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty-stratified evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_by_difficulty(model, X, y, meta):
    """
    Break down evaluation metrics by difficulty level.

    Args:
        model: fitted sklearn pipeline
        X:     feature matrix
        y:     ground-truth labels
        meta:  list of per-pair metadata dicts (must contain 'difficulty')

    Returns:
        dict mapping difficulty → metric dict
    """
    difficulties = sorted({m.get("difficulty", "unknown") for m in meta})
    results = {}

    for diff in difficulties:
        idx = [i for i, m in enumerate(meta) if m.get("difficulty") == diff]
        if not idx:
            continue

        idx_arr = np.array(idx)
        X_d = X[idx_arr]
        y_d = y[idx_arr]

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_d)[:, 1]
        else:
            y_prob = model.decision_function(X_d)

        y_pred  = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_d, y_pred, y_prob)
        results[diff] = metrics

        print(f"  [{diff:6s}] F1={metrics['f1']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
              f"n={metrics['n_samples']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(model, X, y, meta, threshold: float = 0.5):
    """
    Identify false positives and false negatives with their metadata.

    Returns:
        dict with keys 'false_positives' and 'false_negatives'.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)

    y_pred = (y_prob >= threshold).astype(int)

    false_positives = []
    false_negatives = []

    for i, (true_label, pred_label, prob) in enumerate(zip(y, y_pred, y_prob)):
        entry = {**meta[i], "prob": float(prob), "true_label": int(true_label)}
        if pred_label == 1 and true_label == 0:
            false_positives.append(entry)
        elif pred_label == 0 and true_label == 1:
            false_negatives.append(entry)

    total = len(y)
    print(f"  False positives: {len(false_positives)} / {total} "
          f"({100 * len(false_positives) / total:.1f}%)")
    print(f"  False negatives: {len(false_negatives)} / {total} "
          f"({100 * len(false_negatives) / total:.1f}%)")

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }