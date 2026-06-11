"""
Classifiers for author switch detection.

Provides factory functions for every supported model, a MODEL_REGISTRY dict,
and train_model() / build_model() convenience helpers.

GPU capability is probed once at import time: having CUDA installed does not
guarantee XGBoost or LightGBM were compiled with GPU support, so a tiny
training run is performed to confirm before committing to the GPU code path.

Scaling strategy:
  - Linear/NN models (LR, SVC, MLP): StandardScaler with full mean-centering.
  - Tree models (RF, ET, XGB, LGB): no scaling — trees are scale-invariant.
"""

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")


# ---------------------------------------------------------------------------
# GPU probing (actually run a tiny model to confirm GPU works)
# ---------------------------------------------------------------------------

def _check_xgboost_gpu():
    """Return True only if XGBoost was compiled with GPU support."""
    if not XGBOOST_AVAILABLE:
        return False
    try:
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        xgb.XGBClassifier(tree_method="gpu_hist", n_estimators=1, max_depth=2, verbosity=0).fit(X, y)
        return True
    except Exception:
        return False


def _check_lightgbm_gpu():
    """Return True only if LightGBM was compiled with GPU support."""
    if not LIGHTGBM_AVAILABLE:
        return False
    try:
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        lgb.LGBMClassifier(device="gpu", n_estimators=1, num_leaves=4, verbose=-1).fit(X, y)
        return True
    except Exception:
        return False


_XGBOOST_GPU_AVAILABLE  = _check_xgboost_gpu()
_LIGHTGBM_GPU_AVAILABLE = _check_lightgbm_gpu()

if XGBOOST_AVAILABLE:
    print(f"[classifiers] XGBoost available ({'GPU' if _XGBOOST_GPU_AVAILABLE else 'CPU only'})")
if LIGHTGBM_AVAILABLE:
    print(f"[classifiers] LightGBM available ({'GPU' if _LIGHTGBM_GPU_AVAILABLE else 'CPU only'})")


# ---------------------------------------------------------------------------
# Pipeline with optional scaler
# ---------------------------------------------------------------------------

class OptionalScalerPipeline(Pipeline):
    """
    sklearn Pipeline that supports scaler=None for tree-based models.

    sklearn requires every __init__ parameter to also be a same-named instance
    attribute so get_params() can reflect them. Both scaler and classifier are
    stored explicitly to satisfy this contract.
    """

    def __init__(self, scaler, classifier, memory=None, verbose=False):
        self.scaler     = scaler
        self.classifier = classifier
        steps = [("scale", scaler), ("clf", classifier)] if scaler is not None else [("clf", classifier)]
        super().__init__(steps, memory=memory, verbose=verbose)


def _linear_scaler():
    """StandardScaler compatible with sparse matrices."""
    return StandardScaler(with_mean=False)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_logistic_regression(C=1.0, solver="lbfgs"):
    """
    Logistic Regression baseline.

    Linear decision boundaries are often sufficient for stylometric feature
    spaces, and coefficients provide direct feature-importance interpretability.
    """
    return OptionalScalerPipeline(
        scaler=_linear_scaler(),
        classifier=LogisticRegression(
            C=C, class_weight="balanced", solver=solver, max_iter=1000, random_state=42,
        ),
    )


def make_linear_svc(C=1.0):
    """
    LinearSVC with calibrated probabilities (Platt scaling).

    LinearSVC scales as O(n) in sample count; kernel SVMs are O(n²) and are
    impractical at the tens-of-thousands scale of PAN datasets.
    dual=False is faster when n_features > n_samples.
    """
    return OptionalScalerPipeline(
        scaler=_linear_scaler(),
        classifier=CalibratedClassifierCV(
            LinearSVC(C=C, class_weight="balanced", max_iter=2000, dual=False, random_state=42),
            method="sigmoid",
            cv=5,
        ),
    )


def make_mlp(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256):
    """
    MLP tuned for ~12.5k stylometric input features.

    Default architecture: (256, 128) — large enough to capture feature
    interactions, small enough to avoid overfitting on limited PAN data.
    Early stopping and adaptive learning rate prevent runaway training.
    """
    return OptionalScalerPipeline(
        scaler=_linear_scaler(),
        classifier=MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            batch_size=batch_size,
            random_state=42,
            verbose=True,
        ),
    )


def make_random_forest(n_estimators=500, max_depth=None, calibrate=True):
    """
    Random Forest with optional probability calibration.

    No scaling needed — tree splits are scale-invariant.
    n_jobs=1 avoids multiprocessing fork contention in Docker environments.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    clf = CalibratedClassifierCV(rf, method="sigmoid", cv=3) if calibrate else rf
    return OptionalScalerPipeline(scaler=None, classifier=clf)


def make_extra_trees(n_estimators=300, max_depth=None, calibrate=True):
    """
    Extremely Randomised Trees.

    Random threshold selection typically outperforms Random Forest on mixed
    feature types (ratios, counts, frequencies) and trains faster.
    """
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    clf = CalibratedClassifierCV(et, method="sigmoid", cv=3) if calibrate else et
    return OptionalScalerPipeline(scaler=None, classifier=clf)


def make_xgboost(n_estimators=500, max_depth=6, learning_rate=0.05, use_gpu=True):
    """
    XGBoost gradient boosting with verified GPU detection.

    Falls back to CPU histogram method if GPU support is unavailable in the
    current XGBoost build (CUDA installed ≠ XGBoost compiled for GPU).
    """
    if not XGBOOST_AVAILABLE:
        warnings.warn("XGBoost not installed. Falling back to ExtraTrees.")
        return make_extra_trees()

    if use_gpu and _XGBOOST_GPU_AVAILABLE:
        tree_method, predictor = "gpu_hist", "gpu_predictor"
        print("[classifiers] XGBoost: Using GPU acceleration (verified)")
    else:
        tree_method, predictor = "hist", "cpu_predictor"
        if use_gpu and not _XGBOOST_GPU_AVAILABLE:
            print("[classifiers] XGBoost: GPU not available in build. Using CPU 'hist'.")

    return OptionalScalerPipeline(
        scaler=None,
        classifier=xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            predictor=predictor,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
    )


def make_lightgbm(n_estimators=500, num_leaves=63, learning_rate=0.05, use_gpu=True):
    """
    LightGBM gradient boosting with verified GPU detection.

    Typically the strongest classical model for PAN stylometry tasks.
    Leaf-wise growth and histogram-based splits handle high-dimensional
    stylometric features efficiently even on CPU.
    """
    if not LIGHTGBM_AVAILABLE:
        warnings.warn("LightGBM not installed. Falling back to ExtraTrees.")
        return make_extra_trees()

    if use_gpu and _LIGHTGBM_GPU_AVAILABLE:
        device = "gpu"
        print("[classifiers] LightGBM: Using GPU acceleration (verified)")
    else:
        device = "cpu"
        if use_gpu and not _LIGHTGBM_GPU_AVAILABLE:
            print("[classifiers] LightGBM: GPU not available in build. Using CPU.")

    return OptionalScalerPipeline(
        scaler=None,
        classifier=lgb.LGBMClassifier(
            n_estimators=n_estimators,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            device=device,
            verbose=-1,
            random_state=42,
        ),
    )


# ---------------------------------------------------------------------------
# Model registry and metadata
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "logistic_regression": make_logistic_regression,
    "linear_svc":          make_linear_svc,
    "mlp":                 make_mlp,
    "random_forest":       make_random_forest,
    "extra_trees":         make_extra_trees,
}

if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["xgboost"]  = make_xgboost
if LIGHTGBM_AVAILABLE:
    MODEL_REGISTRY["lightgbm"] = make_lightgbm

MODEL_DESCRIPTIONS = {
    "logistic_regression": "Fast linear baseline, interpretable coefficients",
    "linear_svc":          "Linear SVM with calibrated probabilities",
    "mlp":                 "Neural network (256, 128) tuned for ~12.5k features",
    "random_forest":       "500 trees, calibrated, no scaling needed",
    "extra_trees":         "500 randomised trees, often beats RF",
    "xgboost":             f"Gradient boosting ({'GPU' if _XGBOOST_GPU_AVAILABLE else 'CPU'})",
    "lightgbm":            f"Gradient boosting ({'GPU' if _LIGHTGBM_GPU_AVAILABLE else 'CPU'}), often best",
}

# True = model requires StandardScaler; False = scale-invariant tree model
MODEL_USES_SCALING = {
    "logistic_regression": True,
    "linear_svc":          True,
    "mlp":                 True,
    "random_forest":       False,
    "extra_trees":         False,
    "xgboost":             False,
    "lightgbm":            False,
}

HYPERPARAMETER_GRIDS = {
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    },
    "linear_svc": {
        "clf__C": [0.01, 0.1, 1.0],
    },
    "mlp": {
        "clf__hidden_layer_sizes": [(128, 64), (256, 128), (128, 128)],
        "clf__alpha": [1e-4, 1e-3, 1e-2],
        "clf__learning_rate_init": [1e-4, 1e-3],
    },
    "random_forest": {
        "clf__base_estimator__n_estimators": [200, 500],
        "clf__base_estimator__max_depth": [None, 10, 20],
    },
    "extra_trees": {
        "clf__base_estimator__n_estimators": [200, 500],
        "clf__base_estimator__max_depth": [None, 10, 20],
    },
    "xgboost": {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [4, 6, 8],
        "clf__learning_rate": [0.01, 0.05, 0.1],
    },
    "lightgbm": {
        "clf__n_estimators": [200, 500],
        "clf__num_leaves": [31, 63, 127],
        "clf__learning_rate": [0.01, 0.05, 0.1],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(name, **kwargs):
    """Instantiate and return the named model (unfitted)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def _get_hyperparameter_space(name):
    if name not in HYPERPARAMETER_GRIDS:
        raise KeyError(
            f"No hyperparameter search space defined for model '{name}'. "
            "Add one to HYPERPARAMETER_GRIDS in models/classifiers.py."
        )
    return HYPERPARAMETER_GRIDS[name]


def search_model(
    name,
    X_train,
    y_train,
    method: str = "randomized",
    cv: int = 5,
    n_iter: int = 25,
    scoring: str = "f1",
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: bool = False,
    model_kwargs: dict = None,
):
    """Search for the best hyperparameters and return the fitted estimator."""
    model_kwargs = model_kwargs or {}
    param_space = _get_hyperparameter_space(name)
    estimator = build_model(name, **model_kwargs)

    if method == "grid":
        search = GridSearchCV(
            estimator,
            param_space,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            verbose=2 if verbose else 0,
        )
    elif method == "randomized":
        search = RandomizedSearchCV(
            estimator,
            param_space,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
            verbose=1 if verbose else 0,
        )
    else:
        raise ValueError("method must be 'grid' or 'randomized'")

    print(f"[classifiers] Running {method} hyperparameter search for {name}...")
    search.fit(X_train, y_train)
    print(f"[classifiers] Best {name} params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search


def train_model(
    name,
    X_train,
    y_train,
    use_hyperparam_search: bool = False,
    search_method: str = "grid",
    search_cv: int = 5,
    search_n_iter: int = 25,
    search_scoring: str = "f1",
    search_verbose: bool = False,
    **kwargs,
):
    """Instantiate, fit, and return the named model."""
    if use_hyperparam_search:
        model, best_params, _ = search_model(
            name,
            X_train,
            y_train,
            method=search_method,
            cv=search_cv,
            n_iter=search_n_iter,
            scoring=search_scoring,
            verbose=search_verbose,
            model_kwargs=kwargs,
        )
    else:
        model = build_model(name, **kwargs)
        model.fit(X_train, y_train)

    uses_scaling = MODEL_USES_SCALING.get(name, True)
    print(
        f"[classifiers] Trained '{name}' on {len(y_train)} samples "
        f"({X_train.shape[1]} features, scaling={'yes' if uses_scaling else 'no'})"
    )

    if name == "mlp":
        clf = model
        if hasattr(model, "named_steps") and "clf" in model.named_steps:
            clf = model.named_steps["clf"]
        if hasattr(clf, "loss_curve_"):
            loss_curve = clf.loss_curve_
            print(
                f"[classifiers] MLP iterations: {len(loss_curve)}; "
                f"final loss = {loss_curve[-1]:.5f}"
            )
        if hasattr(clf, "n_iter_"):
            print(f"[classifiers] MLP n_iter_ = {clf.n_iter_}")

    return model


def list_available_models():
    """Print all available models with metadata."""
    print("\n" + "=" * 70)
    print("Available Models for Author Switch Detection")
    print("=" * 70)
    for name, description in MODEL_DESCRIPTIONS.items():
        available = "✓" if name in MODEL_REGISTRY else "✗"
        scaling   = "scaled" if MODEL_USES_SCALING.get(name, True) else "no scaling"
        print(f"  [{available}] {name:25s} [{scaling:10s}] {description}")
    print("=" * 70)
    if XGBOOST_AVAILABLE:
        print(f"  XGBoost GPU:  {'Available ✓' if _XGBOOST_GPU_AVAILABLE else 'Not available ✗'}")
    if LIGHTGBM_AVAILABLE:
        print(f"  LightGBM GPU: {'Available ✓' if _LIGHTGBM_GPU_AVAILABLE else 'Not available ✗'}")
    print()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(1000, 12500).astype(np.float32)
    y = (np.random.rand(1000) > 0.7).astype(np.int32)
    print(f"Test data: {X.shape}, class balance: {np.mean(y):.2%} positive\n")
    for name in MODEL_REGISTRY:
        try:
            model = train_model(name, X[:800], y[:800])
            score = model.score(X[800:], y[800:])
            print(f"  {name:25s} Test accuracy: {score:.3f}")
        except Exception as e:
            print(f"  {name:25s} Failed: {str(e)[:80]}")
    list_available_models()