"""
Feature importance analysis for author switch detection.
Updated to handle OptionalScalerPipeline with tree models (no scaling).
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance as _sklearn_perm


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic)
# ---------------------------------------------------------------------------

def permutation_importance(model, X_val, y_val, feature_names=None,
                            n_repeats=10, top_k=20, random_state=42):
    """
    Rank features by F1 drop when each is randomly shuffled.
    
    Works for any fitted model (trees, linear, neural nets).
    Handles OptionalScalerPipeline correctly.
    
    A large drop → the model depends heavily on this feature.
    Near zero or negative → the feature is ignored or adds noise.
    
    Args:
        model: fitted sklearn Pipeline or estimator
        X_val: validation features
        y_val: validation labels
        feature_names: list of feature names (auto-generated if None)
        n_repeats: number of times to shuffle each feature
        top_k: number of top features to display
        random_state: random seed for reproducibility
    
    Returns:
        list of dicts with feature importance info
    """
    # Handle OptionalScalerPipeline - get the classifier
    clf = _get_classifier(model)
    
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
    
    print(f"\n{'='*70}")
    print(f"Permutation Importance (top {top_k} features)")
    print(f"{'='*70}")
    print(f"{'Feature':<40} {'Importance':>12} {'Std':>10}")
    print(f"{'-'*70}")
    
    for name, imp, std in ranked[:top_k]:
        bar = "█" * min(30, max(0, int(imp * 200)))
        print(f"  {name:<38} {imp:>8.4f} ± {std:.4f}  {bar}")
    
    # Summary statistics
    positive_features = sum(1 for _, imp, _ in ranked if imp > 0)
    print(f"\n  {positive_features}/{n_feats} features have positive importance")
    print(f"{'='*70}\n")
    
    return [
        {"name": n, "importance": round(float(imp), 6), "std": round(float(std), 6)}
        for n, imp, std in ranked
    ]


# ---------------------------------------------------------------------------
# Coefficient analysis (Logistic Regression only)
# ---------------------------------------------------------------------------

def logistic_coefficients(model, feature_names=None, top_k=20):
    """
    Extract and display the top positive and negative LR coefficients.
    
    Positive coefficients → feature pushes prediction towards class 1 (switch).
    Negative coefficients → feature pushes prediction towards class 0 (same author).
    
    Handles OptionalScalerPipeline correctly.
    
    Args:
        model: fitted sklearn Pipeline with LogisticRegression
        feature_names: list of feature names
        top_k: number of top features to display
    
    Returns:
        list of dicts with coefficient info, or None if not applicable
    """
    # Navigate pipeline to find the classifier
    clf = _get_classifier(model)
    
    if not hasattr(clf, "coef_"):
        print("[importance] Model doesn't have coefficients — skipping coefficient analysis.")
        print("[importance] Use permutation_importance() for model-agnostic analysis.")
        return None
    
    coef = clf.coef_[0]
    n_feats = len(coef)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_feats)]
    
    # Sort by absolute coefficient magnitude
    pairs = sorted(zip(feature_names, coef), key=lambda x: -abs(x[1]))
    
    print(f"\n{'='*70}")
    print(f"Logistic Regression Coefficients (top {top_k})")
    print(f"{'='*70}")
    print(f"  {'Feature':<38} {'Coefficient':>12}  Direction")
    print(f"  {'-'*70}")
    
    for name, c in pairs[:top_k]:
        direction = "→ SWITCH" if c > 0 else "→ SAME  "
        print(f"  {name:<38} {c:>12.4f}  {direction}")
    
    # Distribution stats
    pos_coef = sum(1 for _, c in pairs if c > 0)
    neg_coef = sum(1 for _, c in pairs if c < 0)
    print(f"\n  {pos_coef} positive coefficients (→ SWITCH)")
    print(f"  {neg_coef} negative coefficients (→ SAME)")
    print(f"{'='*70}\n")
    
    return [
        {"name": n, "coefficient": round(float(c), 6)}
        for n, c in pairs
    ]


# ---------------------------------------------------------------------------
# SHAP analysis (optional)
# ---------------------------------------------------------------------------

def shap_summary(model, X_background, X_explain, feature_names=None, 
                 max_display=20, use_kernel=False):
    """
    Compute SHAP values using appropriate explainer.
    
    Handles OptionalScalerPipeline correctly by:
    1. Extracting the classifier
    2. Applying any scaler transform if present
    3. Choosing the right SHAP explainer
    
    Args:
        model: fitted sklearn Pipeline
        X_background: background data for explainer
        X_explain: data to explain
        feature_names: feature names
        max_display: max features to show
        use_kernel: force KernelExplainer even for linear models
    
    Returns:
        list of dicts with SHAP importance, or None if shap not installed
    """
    try:
        import shap
    except ImportError:
        print("[importance] 'shap' not installed. Run: pip install shap")
        return None
    
    # Get classifier and handle scaling
    clf = _get_classifier(model)
    
    # Check if pipeline has a scaler and apply it
    if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
        scaler = model.named_steps['scale']
        if scaler is not None:
            X_bg = scaler.transform(X_background)
            X_exp = scaler.transform(X_explain)
        else:
            X_bg, X_exp = X_background, X_explain
    else:
        X_bg, X_exp = X_background, X_explain
    
    # Choose explainer
    is_linear = hasattr(clf, "coef_") and not use_kernel
    
    if is_linear:
        print("[importance] Using LinearExplainer (model is linear)")
        explainer = shap.LinearExplainer(clf, X_bg)
    else:
        print("[importance] Using KernelExplainer (may be slow)")
        # Subsample background if large
        if len(X_bg) > 100:
            print(f"[importance] Subsampling background from {len(X_bg)} to 100")
            X_bg = shap.kmeans(X_bg, 100)
        explainer = shap.KernelExplainer(
            clf.predict_proba if hasattr(clf, 'predict_proba') else clf.decision_function,
            X_bg
        )
    
    # Compute SHAP values
    print(f"[importance] Computing SHAP values for {len(X_exp)} samples...")
    shap_values = explainer.shap_values(X_exp)
    
    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (switch)
    
    # Summary plot
    shap.summary_plot(
        shap_values, X_exp,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    
    # Calculate mean absolute SHAP values
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        zip(feature_names or range(len(mean_abs)), mean_abs),
        key=lambda x: -x[1]
    )
    
    print(f"\n{'='*70}")
    print(f"SHAP Feature Importance (top {min(max_display, len(ranked))})")
    print(f"{'='*70}")
    for name, v in ranked[:max_display]:
        bar = "█" * min(30, int(v * 200))
        print(f"  {str(name):<38} {v:>8.4f}  {bar}")
    print(f"{'='*70}\n")
    
    return [
        {"name": str(n), "mean_abs_shap": round(float(v), 6)}
        for n, v in ranked
    ]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def _get_classifier(model):
    """
    Extract the actual classifier from a pipeline.
    Handles OptionalScalerPipeline, CalibratedClassifierCV, etc.
    """
    # Try named steps (sklearn Pipeline)
    if hasattr(model, 'named_steps'):
        clf = model.named_steps.get('clf', model)
        # Unwrap CalibratedClassifierCV
        if hasattr(clf, 'calibrated_classifiers_'):
            return clf.estimator
        return clf
    return model


def get_feature_importance_dataframe(importance_results, top_k=None, sort_by='importance'):
    """
    Convert importance results to a pandas DataFrame for further analysis.
    
    Args:
        importance_results: output from permutation_importance or logistic_coefficients
        top_k: limit to top K features
        sort_by: column to sort by
    
    Returns:
        pandas DataFrame
    """
    import pandas as pd
    
    df = pd.DataFrame(importance_results)
    
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    if top_k:
        df = df.head(top_k)
    
    return df