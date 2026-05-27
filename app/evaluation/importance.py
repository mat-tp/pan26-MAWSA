"""
Feature importance analysis.

Three complementary approaches:

1. Permutation importance (model-agnostic):
   Randomly shuffle one feature column, measure F1 drop.
   Works for any fitted model.

2. Coefficient analysis (Logistic Regression only):
   Coefficients reveal which features push the decision boundary
   towards "switch" vs "same author".

3. SHAP values (optional, requires shap package):
   Provides per-prediction, per-feature explanations.
   Use LinearExplainer for LR (fast), KernelExplainer for others (slow).
"""

import numpy as np
from sklearn.inspection import permutation_importance as _sklearn_perm


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic)
# ---------------------------------------------------------------------------

def permutation_importance(model, X_val, y_val, feature_names=None,
                            n_repeats=10, top_k=20):
    """
    Rank features by F1 drop when each is randomly shuffled.

    A large drop → the model depends heavily on this feature.
    Near zero or negative → the feature is ignored or adds noise.
    """
    result = _sklearn_perm(
        model, X_val, y_val,
        n_repeats=n_repeats,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
    )

    n_feats = len(result.importances_mean)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_feats)]

    ranked = sorted(
        zip(feature_names, result.importances_mean, result.importances_std),
        key=lambda x: -x[1],
    )

    print(f"\nPermutation importance (top {top_k}):")
    for name, imp, std in ranked[:top_k]:
        bar = "█" * max(0, int(imp * 200))
        print(f"  {name:<35} {imp:>8.4f} ± {std:.4f}  {bar}")

    return [
        {"name": n, "importance": round(float(imp), 6), "std": round(float(std), 6)}
        for n, imp, std in ranked
    ]


# ---------------------------------------------------------------------------
# Coefficient analysis (Logistic Regression)
# ---------------------------------------------------------------------------

def logistic_coefficients(model, feature_names=None, top_k=20):
    """
    Extract and display the top positive and negative LR coefficients.

    Positive coefficients → feature pushes prediction towards class 1 (switch).
    Negative coefficients → feature pushes prediction towards class 0 (same author).
    """
    # Navigate past the Pipeline wrapper to the LogisticRegression step
    clf = model.named_steps.get("clf", model)
    if not hasattr(clf, "coef_"):
        print("[importance] Model is not a LogisticRegression — skipping.")
        return None

    coef = clf.coef_[0]
    n_feats = len(coef)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_feats)]

    pairs = sorted(zip(feature_names, coef), key=lambda x: -abs(x[1]))

    print(f"\nTop {top_k} Logistic Regression coefficients:")
    print(f"  {'Feature':<35} {'Coeff':>10}  Direction")
    print("  " + "-" * 60)
    for name, c in pairs[:top_k]:
        direction = "→ SWITCH" if c > 0 else "→ SAME  "
        print(f"  {name:<35} {c:>10.4f}  {direction}")

    return [
        {"name": n, "coefficient": round(float(c), 6)}
        for n, c in pairs
    ]


# ---------------------------------------------------------------------------
# SHAP analysis (optional)
# ---------------------------------------------------------------------------

def shap_summary(model, X_background, X_explain, feature_names=None, max_display=20):
    """
    Compute SHAP values for X_explain using a background sample X_background.

    Requires the 'shap' package:  pip install shap

    For Logistic Regression use LinearExplainer (fast).
    For all others use KernelExplainer (slow — subsample X_background if needed).
    """
    try:
        import shap
    except ImportError:
        print("[importance] 'shap' not installed. Run: pip install shap")
        return None

    clf = model.named_steps.get("clf", model)
    scaler = model.named_steps.get("scaler", None)

    # Transform inputs through the scaler if present
    if scaler is not None:
        X_bg  = scaler.transform(X_background)
        X_exp = scaler.transform(X_explain)
    else:
        X_bg, X_exp = X_background, X_explain

    if hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, X_bg)
    else:
        explainer = shap.KernelExplainer(
            clf.predict_proba, shap.kmeans(X_bg, 50)
        )

    shap_values = explainer.shap_values(X_exp)

    # For binary classifiers shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print("\nSHAP summary (mean |SHAP| per feature, top features):")
    shap.summary_plot(
        shap_values, X_exp,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(zip(feature_names or range(len(mean_abs)), mean_abs),
                    key=lambda x: -x[1])

    return [
        {"name": str(n), "mean_abs_shap": round(float(v), 6)}
        for n, v in ranked
    ]
