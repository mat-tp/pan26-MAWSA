"""
Feature importance analysis for author switch detection.

Provides permutation importance, logistic regression coefficients, and
optional SHAP values. All plots are saved to PLOTS_DIR automatically.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance as _sklearn_perm


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[importance] Plot saved → {path}")


def _get_style() -> str:
    available = plt.style.available
    for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot", "default"):
        if candidate in available:
            return candidate
    return "default"


def _get_classifier(model):
    """Extract the actual classifier from a pipeline."""
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
        if hasattr(clf, "calibrated_classifiers_"):
            return clf.estimator
        return clf
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Permutation importance
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(
    model,
    X_val,
    y_val,
    feature_names=None,
    n_repeats: int = 10,
    top_k: int = 20,
    random_state: int = 42,
    save_plot: bool = True,
):
    """
    Rank features by F1 drop when each is randomly shuffled.

    A large drop → the model depends heavily on this feature.
    Near zero or negative → the feature is ignored or adds noise.

    Args:
        model:         fitted sklearn Pipeline or estimator
        X_val:         validation features
        y_val:         validation labels
        feature_names: list of feature names (auto-generated if None)
        n_repeats:     number of times to shuffle each feature
        top_k:         number of top features to display / plot
        random_state:  RNG seed
        save_plot:     whether to save a PNG bar chart

    Returns:
        list of dicts [{name, importance, std}, ...]  sorted descending
    """
    result = _sklearn_perm(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring="f1",
        random_state=random_state,
        n_jobs=-1,
    )

    n_feats = len(result.importances_mean)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_feats)]

    ranked = sorted(
        zip(feature_names, result.importances_mean, result.importances_std),
        key=lambda x: -x[1],
    )

    # Console summary
    print(f"\n{'=' * 70}")
    print(f"Permutation Importance (top {top_k} features)")
    print(f"{'=' * 70}")
    print(f"{'Feature':<40} {'Importance':>12} {'Std':>10}")
    print(f"{'-' * 70}")
    for name, imp, std in ranked[:top_k]:
        bar = "█" * min(30, max(0, int(imp * 200)))
        print(f"  {name:<38} {imp:>8.4f} ± {std:.4f}  {bar}")

    positive_features = sum(1 for _, imp, _ in ranked if imp > 0)
    print(f"\n  {positive_features}/{n_feats} features have positive importance")
    print(f"{'=' * 70}\n")

    # Save plot
    if save_plot:
        _plot_importance_bars(ranked, top_k, title="Permutation Importance (top features)")

    return [
        {"name": n, "importance": round(float(imp), 6), "std": round(float(std), 6)}
        for n, imp, std in ranked
    ]


def _plot_importance_bars(
    ranked: list,
    top_k: int,
    title: str = "Feature Importance",
    path: str = None,
    dpi: int = 150,
) -> None:
    """Horizontal bar chart of feature importances, saved to PLOTS_DIR."""
    from utils.config import PLOT_IMPORTANCE_PATH, PLOT_DPI

    path = path or PLOT_IMPORTANCE_PATH
    dpi  = dpi  or PLOT_DPI

    top = ranked[:top_k]
    if not top:
        return

    names   = [t[0] for t in top][::-1]
    values  = [t[1] for t in top][::-1]
    errors  = [t[2] for t in top][::-1] if len(top[0]) > 2 else None

    with plt.style.context(_get_style()):
        fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.30)))
        colors = ["#4878CF" if v >= 0 else "#D65F5F" for v in values]
        ax.barh(range(len(names)), values,
                xerr=errors, color=colors, alpha=0.85, capsize=3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Importance (F1 drop)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.35)

    _save_fig(fig, path, dpi=dpi)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression coefficients
# ─────────────────────────────────────────────────────────────────────────────

def logistic_coefficients(
    model,
    feature_names=None,
    top_k: int = 20,
    save_plot: bool = True,
):
    """
    Extract and display top positive and negative LR coefficients.

    Returns:
        list of dicts [{name, coefficient}, ...]  sorted by |coeff| descending,
        or None if not applicable.
    """
    from utils.config import PLOT_IMPORTANCE_PATH, PLOT_DPI

    clf = _get_classifier(model)

    if not hasattr(clf, "coef_"):
        print("[importance] Model doesn't have coefficients — skipping.")
        return None

    coef = clf.coef_[0]
    n_feats = len(coef)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_feats)]

    pairs = sorted(zip(feature_names, coef), key=lambda x: -abs(x[1]))

    print(f"\n{'=' * 70}")
    print(f"Logistic Regression Coefficients (top {top_k})")
    print(f"{'=' * 70}")
    print(f"  {'Feature':<38} {'Coefficient':>12}  Direction")
    print(f"  {'-' * 70}")
    for name, c in pairs[:top_k]:
        direction = "→ SWITCH" if c > 0 else "→ SAME  "
        print(f"  {name:<38} {c:>12.4f}  {direction}")

    pos_coef = sum(1 for _, c in pairs if c > 0)
    neg_coef = sum(1 for _, c in pairs if c < 0)
    print(f"\n  {pos_coef} positive coefficients (→ SWITCH)")
    print(f"  {neg_coef} negative coefficients (→ SAME)")
    print(f"{'=' * 70}\n")

    if save_plot:
        coeff_path = PLOT_IMPORTANCE_PATH.replace(".png", "_coeff.png")
        ranked = [(n, c, 0.0) for n, c in pairs]   # std=0 for coefficients
        _plot_importance_bars(
            ranked, top_k,
            title="LR Coefficients (top features by |coeff|)",
            path=coeff_path,
        )

    return [{"name": n, "coefficient": round(float(c), 6)} for n, c in pairs]


# ─────────────────────────────────────────────────────────────────────────────
# SHAP (optional)
# ─────────────────────────────────────────────────────────────────────────────

def shap_summary(model, X_background, X_explain, feature_names=None,
                 max_display: int = 20, use_kernel: bool = False):
    """
    Compute SHAP values using the appropriate explainer.

    Returns:
        list of dicts [{name, mean_abs_shap}, ...], or None if shap not installed.
    """
    try:
        import shap
    except ImportError:
        print("[importance] 'shap' not installed. Run: pip install shap")
        return None

    clf = _get_classifier(model)

    # Apply scaler if present
    if hasattr(model, "named_steps") and "scale" in model.named_steps:
        scaler = model.named_steps["scale"]
        if scaler is not None:
            X_bg  = scaler.transform(X_background)
            X_exp = scaler.transform(X_explain)
        else:
            X_bg, X_exp = X_background, X_explain
    else:
        X_bg, X_exp = X_background, X_explain

    is_linear = hasattr(clf, "coef_") and not use_kernel

    if is_linear:
        print("[importance] Using LinearExplainer")
        explainer = shap.LinearExplainer(clf, X_bg)
    else:
        print("[importance] Using KernelExplainer (may be slow)")
        if len(X_bg) > 100:
            print(f"[importance] Subsampling background from {len(X_bg)} to 100")
            X_bg = shap.kmeans(X_bg, 100)
        pred_fn = (
            clf.predict_proba
            if hasattr(clf, "predict_proba")
            else clf.decision_function
        )
        explainer = shap.KernelExplainer(pred_fn, X_bg)

    print(f"[importance] Computing SHAP values for {len(X_exp)} samples...")
    shap_values = explainer.shap_values(X_exp)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(feature_names or range(len(mean_abs)), mean_abs),
        key=lambda x: -x[1],
    )

    print(f"\n{'=' * 70}")
    print(f"SHAP Feature Importance (top {min(max_display, len(ranked))})")
    print(f"{'=' * 70}")
    for name, v in ranked[:max_display]:
        bar = "█" * min(30, int(v * 200))
        print(f"  {str(name):<38} {v:>8.4f}  {bar}")
    print(f"{'=' * 70}\n")

    return [{"name": str(n), "mean_abs_shap": round(float(v), 6)} for n, v in ranked]


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame helper
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_importance_dataframe(importance_results, top_k=None, sort_by="importance"):
    """Convert importance results to a pandas DataFrame."""
    import pandas as pd

    df = pd.DataFrame(importance_results)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    if top_k:
        df = df.head(top_k)
    return df