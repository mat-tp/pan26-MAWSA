"""
Evaluation: cross-validation, held-out metrics, per-difficulty breakdown.

All evaluation functions return plain dicts so results can be saved as JSON
or CSV without custom serialisation logic.
"""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)

from app.models.classifiers import MODEL_REGISTRY

# Models that must be skipped above a row threshold (OOM / hang risk)
_EXPENSIVE_MODELS = {
    "svm": 5_000,  # RBF kernel: O(n²) memory — should never reach here if
    # config is correct, but guard stays as a safety net
    "knn": 20_000,  # brute-force KNN: slow at predict time on large datasets
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(model, X, y, groups, n_splits=5):
    """
    Stratified group k-fold cross-validation scored by F1.

    StratifiedGroupKFold ensures:
      - class balance is preserved across folds (stratified),
      - sentence pairs from the same problem stay in the same fold (grouped).
    The second point prevents data leakage: if two pairs from the same
    document appear in both train and test, the model gets an unfair hint.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="f1",
        return_train_score=True,
        n_jobs=-1,
    )
    return {
        "test_f1_mean": round(float(np.mean(scores["test_score"])), 4),
        "test_f1_std": round(float(np.std(scores["test_score"])), 4),
        "train_f1_mean": round(float(np.mean(scores["train_score"])), 4),
        "train_f1_std": round(float(np.std(scores["train_score"])), 4),
    }


def compare_all_models(X, y, groups, n_splits=5):
    """Run cross-validation for every model in MODEL_REGISTRY."""
    n_rows = X.shape[0]
    results = {}

    for name, factory in MODEL_REGISTRY.items():
        limit = _EXPENSIVE_MODELS.get(name)
        if limit and n_rows > limit:
            print(
                f"[evaluation] SKIPPING {name}: {n_rows} rows exceeds "
                f"safe limit of {limit}. See _EXPENSIVE_MODELS in metrics.py."
            )
            results[name] = {
                "skipped": True,
                "reason": f"n_rows={n_rows} > limit={limit}",
            }
            continue

        print(f"[evaluation] Cross-validating: {name} ...")
        res = cross_validate_model(factory(), X, y, groups, n_splits=n_splits)
        results[name] = res
        print(f"  F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------


def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Compute and print precision, recall, F1, AUC, and confusion matrix.
    Returns a dict — safe to serialise with json.dump().
    """
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    rep = classification_report(
        y_test,
        y_pred,
        target_names=["same_author", "switch"],
        zero_division=0,
    )

    # AUC requires probability estimates
    auc = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 1]
            auc = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception:
            pass

    print(f"\n{'=' * 55}\n  {model_name}\n{'=' * 55}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    if auc is not None:
        print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n{rep}")
    print(f"Confusion matrix:\n{cm}\n")

    return {
        "model": model_name,
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "auc": auc,
        "confusion_matrix": cm,
        "report": rep,
    }


# ---------------------------------------------------------------------------
# Per-difficulty breakdown
# ---------------------------------------------------------------------------


def evaluate_by_difficulty(model, X_test, y_test, meta):
    """
    Break down F1 by difficulty level (easy / medium / hard).

    meta is the list of dicts returned by build_pairwise_dataset().
    """
    difficulties = sorted(set(m["difficulty"] for m in meta))
    results = {}

    for diff in difficulties:
        idx = [i for i, m in enumerate(meta) if m["difficulty"] == diff]
        if not idx:
            continue
        X_d = X_test[idx]
        y_d = y_test[idx]
        y_pred = model.predict(X_d)

        results[diff] = {
            "n_pairs": len(idx),
            "n_switches": int(sum(y_d)),
            "f1": round(float(f1_score(y_d, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_d, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_d, y_pred, zero_division=0)), 4),
        }
        print(
            f"  [{diff}] F1={results[diff]['f1']:.4f}  "
            f"P={results[diff]['precision']:.4f}  "
            f"R={results[diff]['recall']:.4f}  "
            f"(n={len(idx)})"
        )

    return results


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------


def error_analysis(model, X_test, y_test, meta):
    """
    Return false positives and false negatives with their metadata.

    Useful for inspecting which sentence pairs are hardest for the model.
    """
    y_pred = model.predict(X_test)
    false_pos, false_neg = [], []

    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if pred == 1 and true == 0:
            false_pos.append(meta[i])
        elif pred == 0 and true == 1:
            false_neg.append(meta[i])

    print(f"  False positives: {len(false_pos)}")
    print(f"  False negatives: {len(false_neg)}")
    return {"false_positives": false_pos, "false_negatives": false_neg}
