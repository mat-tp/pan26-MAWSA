import os
os.environ.setdefault("MPLBACKEND", "Agg")
"""
Cross-validation and evaluation metrics for author switch detection.

Provides robust cross-validation with group-aware splitting to prevent
data leakage between problems, plus comprehensive evaluation metrics
for threshold optimization and model comparison.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)


def cross_validate_model(model, X, y, groups, n_splits=5, scoring=None, 
                         return_models=False, threshold=0.5):
    """
    Perform group-aware cross-validation to prevent data leakage.
    
    Groups = problem IDs ensure all pairs from the same document stay together.
    
    Args:
        model: sklearn Pipeline or estimator
        X: feature matrix
        y: labels
        groups: group labels (problem indices)
        n_splits: number of CV folds
        scoring: metrics to compute (default: comprehensive set)
        return_models: if True, return fitted models for each fold
        threshold: classification threshold for metrics
    
    Returns:
        dict with mean/std for each metric
    """
    if scoring is None:
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'average_precision': 'average_precision',
        }
    
    # Handle optional scaler in pipelines
    if hasattr(model, 'named_steps'):
        if 'scaler' in model.named_steps and model.named_steps['scaler'] is None:
            print("[metrics] Note: Model uses no scaling (tree-based)")
    
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            groups=groups,
            scoring=scoring,
            return_estimator=return_models,
            n_jobs=-1,
            error_score='raise',
        )
    
    # Aggregate results
    results = {}
    for key in scoring:
        scores = cv_results[f'test_{key}']
        results[f'test_{key}_mean'] = np.mean(scores)
        results[f'test_{key}_std'] = np.std(scores)
    
    # Add train scores for overfitting detection
    if 'train_f1' in cv_results:
        results['train_f1_mean'] = np.mean(cv_results['train_f1'])
        results['train_f1_std'] = np.std(cv_results['train_f1'])
    
    # Store models if requested
    if return_models:
        results['models'] = cv_results['estimator']
    
    # Print summary
    print(f"\n[metrics] {n_splits}-fold CV results (group-aware):")
    print(f"  F1:          {results['test_f1_mean']:.4f} ± {results['test_f1_std']:.4f}")
    print(f"  ROC-AUC:     {results['test_roc_auc_mean']:.4f} ± {results['test_roc_auc_std']:.4f}")
    print(f"  Avg Prec:    {results['test_average_precision_mean']:.4f} ± {results['test_average_precision_std']:.4f}")
    print(f"  Accuracy:    {results['test_accuracy_mean']:.4f} ± {results['test_accuracy_std']:.4f}")
    print(f"  Precision:   {results['test_precision_mean']:.4f} ± {results['test_precision_std']:.4f}")
    print(f"  Recall:      {results['test_recall_mean']:.4f} ± {results['test_recall_std']:.4f}")
    
    # Overfitting check
    if 'train_f1_mean' in results:
        gap = results['train_f1_mean'] - results['test_f1_mean']
        if gap > 0.05:
            print(f"  ⚠ Overfitting detected: train-test F1 gap = {gap:.4f}")
    
    return results


def evaluate_model(model, X_test, y_test, threshold=0.5, plot=True):
    """
    Comprehensive evaluation on a test set.
    
    Args:
        model: fitted sklearn Pipeline
        X_test: test features
        y_test: test labels
        threshold: classification threshold
        plot: generate diagnostic plots
    
    Returns:
        dict of metrics
    """
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    
    y_pred = (y_prob >= threshold).astype(int)
    
    # Compute all metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)
    
    # Print report
    print_classification_report(y_test, y_pred, y_prob, metrics)
    
    # Optional plots
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        plot_confusion_matrix(y_test, y_pred, ax=axes[0])
        plot_roc_curve(y_test, y_prob, metrics['roc_auc'], ax=axes[1])
        plot_precision_recall_curve(y_test, y_prob, metrics['avg_precision'], ax=axes[2])
        plt.tight_layout()
        plt.show()
    
    return metrics


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_prob: predicted probabilities (optional, for AUC)
    
    Returns:
        dict of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['avg_precision'] = average_precision_score(y_true, y_prob)
    
    # Class distribution
    metrics['n_samples'] = len(y_true)
    metrics['n_positive'] = int(np.sum(y_true))
    metrics['n_negative'] = int(np.sum(y_true == 0))
    metrics['positive_rate'] = np.mean(y_true)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics.update({
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
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
    print(f"  Negatives:    {metrics['n_negative']} ({1-metrics['positive_rate']:.1%})")
    print("-" * 60)
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
    if 'avg_precision' in metrics:
        print(f"  Avg Precision:{metrics['avg_precision']:.4f}")
    print("-" * 60)
    print(f"  Confusion Matrix:")
    print(f"    TN={metrics['true_negatives']:5d}  FP={metrics['false_positives']:5d}")
    print(f"    FN={metrics['false_negatives']:5d}  TP={metrics['true_positives']:5d}")
    print("=" * 60)


def find_optimal_threshold(model, X_val, y_val, metric='f1', plot=True):
    """
    Find optimal classification threshold.
    
    Args:
        model: fitted model with predict_proba
        X_val: validation features
        y_val: validation labels
        metric: metric to optimize ('f1', 'precision', 'recall')
        plot: visualize threshold sweep
    
    Returns:
        optimal_threshold, best_metric_value
    """
    if not hasattr(model, 'predict_proba'):
        print("[metrics] Model doesn't support predict_proba. Using default threshold 0.5")
        return 0.5, None
    
    y_prob = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_val, y_pred)
        elif metric == 'precision':
            score = precision_score(y_val, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_val, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(thresholds, scores, 'b-', linewidth=2)
        ax.axvline(best_threshold, color='r', linestyle='--', 
                   label=f'Best threshold: {best_threshold:.3f} ({metric}={best_score:.3f})')
        ax.axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
        ax.set_xlabel('Threshold')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'Threshold Optimization ({metric.upper()})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
    
    print(f"\n[metrics] Optimal threshold: {best_threshold:.3f} ({metric}={best_score:.4f})")
    return best_threshold, best_score


def plot_confusion_matrix(y_true, y_pred, ax=None, normalize=True):
    """Plot confusion matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix' + (' (normalized)' if normalize else ''))
    plt.colorbar(im, ax=ax)
    
    classes = ['Same Author', 'Switch']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return ax


def plot_roc_curve(y_true, y_prob, auc_score=None, ax=None):
    """Plot ROC curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, 
            label=f'ROC (AUC = {auc_score:.3f})' if auc_score else 'ROC')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return ax


def plot_precision_recall_curve(y_true, y_prob, avg_precision=None, ax=None):
    """Plot precision-recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    ax.plot(recall, precision, 'b-', linewidth=2,
            label=f'PR (AP = {avg_precision:.3f})' if avg_precision else 'PR')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    # Add baseline
    baseline = np.mean(y_true)
    ax.axhline(baseline, color='r', linestyle='--', 
               label=f'Baseline ({baseline:.3f})')
    ax.legend()
    return ax

# ─────────────────────────────────────────────────────────────────────────────
# Multi-model comparison (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def compare_all_models(X, y, groups, n_splits=5):
    """
    Cross-validate every model in MODELS_TO_COMPARE and return a summary dict.

    Models are evaluated sequentially to avoid holding multiple copies of the
    CV splits in memory at once.

    Returns:
        dict mapping model_name → CV result dict
    """
    from utils.config import MODELS_TO_COMPARE, RANDOM_SEED
    from models.classifiers import build_model

    results = {}
    print(f"\n{'='*60}")
    print("Model Comparison (cross-validation)")
    print(f"{'='*60}")

    for name in MODELS_TO_COMPARE:
        print(f"\n[metrics] Evaluating: {name}")
        try:
            model = build_model(name)
            cv = cross_validate_model(model, X, y, groups, n_splits=n_splits)
            results[name] = cv
        except Exception as exc:
            print(f"[metrics] {name} failed: {exc}")
            results[name] = {"error": str(exc)}

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"{'Model':<25} {'F1':>8} {'±':>6} {'AUC':>8}")
    print(f"{'-'*60}")
    for name, res in results.items():
        if "error" not in res:
            print(f"  {name:<23} {res['test_f1_mean']:>8.4f} {res['test_f1_std']:>6.4f} "
                  f"{res['test_roc_auc_mean']:>8.4f}")
        else:
            print(f"  {name:<23} ERROR: {res['error'][:30]}")
    print(f"{'='*60}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty-stratified evaluation (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_by_difficulty(model, X, y, meta):
    """
    Break down evaluation metrics by difficulty level (easy / medium / hard).

    Args:
        model:  fitted sklearn pipeline
        X:      feature matrix
        y:      ground-truth labels
        meta:   list of per-pair metadata dicts (must contain 'difficulty')

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

        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_d, y_pred, y_prob)
        results[diff] = metrics

        print(f"  [{diff:6s}] F1={metrics['f1']:.4f}  "
              f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}  "
              f"n={metrics['n_samples']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis (called by main.py)
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(model, X, y, meta, threshold=0.5):
    """
    Identify false positives and false negatives with their metadata.

    Args:
        model:     fitted sklearn pipeline
        X:         feature matrix
        y:         ground-truth labels
        meta:      list of per-pair metadata dicts
        threshold: classification threshold

    Returns:
        dict with keys 'false_positives' and 'false_negatives', each a list of
        metadata dicts annotated with 'prob' and 'true_label'.
    """
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)

    y_pred = (y_prob >= threshold).astype(int)

    false_positives = []
    false_negatives = []

    for i, (true_label, pred_label, prob) in enumerate(zip(y, y_pred, y_prob)):
        if pred_label == 1 and true_label == 0:
            entry = {**meta[i], "prob": float(prob), "true_label": int(true_label)}
            false_positives.append(entry)
        elif pred_label == 0 and true_label == 1:
            entry = {**meta[i], "prob": float(prob), "true_label": int(true_label)}
            false_negatives.append(entry)

    total = len(y)
    print(f"  False positives: {len(false_positives)} / {total} "
          f"({100*len(false_positives)/total:.1f}%)")
    print(f"  False negatives: {len(false_negatives)} / {total} "
          f"({100*len(false_negatives)/total:.1f}%)")

    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }