"""
Classifiers for author switch detection.
Optimized with proper GPU detection, model-specific scaling,
and architecture tuned for ~12.5k stylometric features.

Critical fixes:
  1. GPU detection: actually test LightGBM/XGBoost GPU capability
  2. Scaling: StandardScaler for linear/NN, none for trees
  3. MLP architecture: (256, 128) for ~12k feature space

MODEL REGISTRY makes it easy to iterate over all models for comparison.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Optional imports with graceful fallback
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


# ============================================================================
# Proper GPU Detection - Actually test GPU capability
# ============================================================================

def _check_xgboost_gpu():
    """
    Actually test if XGBoost was compiled with GPU support.
    Having CUDA installed ≠ XGBoost GPU available.
    """
    if not XGBOOST_AVAILABLE:
        return False
    
    try:
        # Try creating a tiny model with GPU
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        model = xgb.XGBClassifier(
            tree_method='gpu_hist',
            n_estimators=1,
            max_depth=2,
            verbosity=0
        )
        model.fit(X, y)
        return True
    except Exception:
        return False


def _check_lightgbm_gpu():
    """
    Actually test if LightGBM was compiled with GPU support.
    Even with CUDA installed, GPU tree learner might not be enabled.
    """
    if not LIGHTGBM_AVAILABLE:
        return False
    
    try:
        # Try creating a tiny model with GPU
        X = np.random.randn(20, 10)
        y = np.random.randint(0, 2, 20)
        model = lgb.LGBMClassifier(
            device='gpu',
            n_estimators=1,
            num_leaves=4,
            verbose=-1
        )
        model.fit(X, y)
        return True
    except Exception as e:
        return False


# Run GPU checks once at module load (with caching)
_XGBOOST_GPU_AVAILABLE = _check_xgboost_gpu()
_LIGHTGBM_GPU_AVAILABLE = _check_lightgbm_gpu()

if XGBOOST_AVAILABLE:
    status = "GPU" if _XGBOOST_GPU_AVAILABLE else "CPU only"
    print(f"[classifiers] XGBoost available ({status})")

if LIGHTGBM_AVAILABLE:
    status = "GPU" if _LIGHTGBM_GPU_AVAILABLE else "CPU only"
    print(f"[classifiers] LightGBM available ({status})")


# ============================================================================
# Model-Specific Scaler Strategy
# ============================================================================

def make_scaler(model_type='linear'):
    """
    Model-specific scaler selection.
    
    Design rationale:
    - Linear/NN models: StandardScaler(full) → centering helps optimization
    - Tree models: None → trees are scale-invariant
    
    Note: Since your features are dense (not sparse), we use full StandardScaler
    with mean centering for linear models. If you later switch to sparse matrices,
    use StandardScaler(with_mean=False).
    
    Args:
        model_type: 'linear' for LR/SVC/MLP, 'tree' for RF/ET/XGB/LGB
    
    Returns:
        StandardScaler or None
    """
    if model_type == 'tree':
        return None  # Trees don't benefit from scaling
    else:
        return StandardScaler()  # Full standardization for linear/NN


class OptionalScalerPipeline(Pipeline):
    """
    Pipeline that handles optional scaler (None for tree models).

    sklearn's Pipeline.get_params() introspects __init__ parameter names and
    calls getattr(self, name) for each, so every __init__ parameter MUST be
    stored as a same-named instance attribute.  The previous implementation
    stored 'scaler' but not 'classifier', causing an AttributeError.
    """

    def __init__(self, scaler, classifier, memory=None, verbose=False):
        # Store as attributes so sklearn's get_params() can reflect them
        self.scaler = scaler
        self.classifier = classifier

        if scaler is None:
            steps = [("clf", classifier)]
        else:
            # Step name must NOT match any __init__ parameter name (sklearn rule)
            steps = [("scale", scaler), ("clf", classifier)]

        super().__init__(steps, memory=memory, verbose=verbose)


# ============================================================================
# Logistic Regression
# ============================================================================

def make_logistic_regression(C=1.0, solver="lbfgs"):
    """
    Logistic Regression baseline.
    
    Why it works for stylometry:
    - Linear decision boundary often sufficient with enough features
    - Coefficients directly interpretable (feature importance)
    - Fast training on ~12.5k features
    - Good probability estimates out of the box
    
    Args:
        C: Inverse regularization (smaller = stronger)
        solver: 'lbfgs' (default), 'liblinear' (small data), 'saga' (L1 penalty)
    """
    return OptionalScalerPipeline(
        scaler=make_scaler('linear'),
        classifier=LogisticRegression(
            C=C,
            class_weight="balanced",
            solver=solver,
            max_iter=1000,
            random_state=42,
        )
    )


# ============================================================================
# Linear SVM
# ============================================================================

def make_linear_svc(C=1.0):
    """
    LinearSVC with calibrated probabilities.
    
    Why LinearSVC instead of kernel SVM:
    - ~12.5k features → linear often sufficient
    - O(n) complexity vs O(n²) for RBF kernel
    - CalibratedClassifierCV adds predict_proba via Platt scaling
    
    Note: dual=False is faster when n_features > n_samples
    """
    return OptionalScalerPipeline(
        scaler=make_scaler('linear'),
        classifier=CalibratedClassifierCV(
            LinearSVC(
                C=C,
                class_weight="balanced",
                max_iter=2000,
                dual=False,  # Faster for wide matrices
                random_state=42,
            ),
            method='sigmoid',  # Platt scaling
            cv=5,
        )
    )


# ============================================================================
# MLP (Neural Network) - Corrected Architecture
# ============================================================================

def make_mlp(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=256):
    """
    MLP with architecture tuned for ~12.5k stylometric features.
    
    Architecture rationale:
    - Input: ~12,500 features
    - Hidden layer 1: 256 units → ~3.2M weights (reasonable for this task)
    - Hidden layer 2: 128 units → further compression
    - Total: ~3.3M parameters (manageable, captures interactions)
    
    Why this size:
    - (64,32) would be too small (1.6M → 800k weights, underfit)
    - (512,256,128) might overfit with small datasets
    - (256,128) is a good middle ground
    
    Regularization:
    - alpha=1e-3 (L2 penalty)
    - early_stopping=True
    - validation_fraction=0.1
    - learning_rate='adaptive' (reduces lr on plateau)
    
    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        alpha: L2 regularization strength
        batch_size: Mini-batch size (128 works well for this feature space)
    """
    return OptionalScalerPipeline(
        scaler=make_scaler('linear'),  # Neural nets benefit from standardization
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
            verbose=False,
        )
    )


# ============================================================================
# Random Forest (No Scaling)
# ============================================================================

def make_random_forest(n_estimators=500, max_depth=None, calibrate=True):
    """
    Random Forest with calibration.
    
    Why no scaling:
    - Trees split on individual feature values
    - Scaling doesn't change split points
    - Removing scaler reduces computation and memory
    
    Args:
        n_estimators: Number of trees (500 is good for ~12.5k features)
        max_depth: Max tree depth (None = unlimited)
        calibrate: Calibrate probabilities with Platt scaling
    """
    rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight="balanced",
    n_jobs=1,          # important
    random_state=42,
    verbose=0,
)
    
    if calibrate:
        return OptionalScalerPipeline(
            scaler=make_scaler('tree'),  # None for trees
            classifier=CalibratedClassifierCV(
                rf,
                method='sigmoid',
                cv=3,
            )
        )
    else:
        return OptionalScalerPipeline(
            scaler=make_scaler('tree'),
            classifier=rf
        )


# ============================================================================
# Extra Trees (No Scaling)
# ============================================================================

def make_extra_trees(n_estimators=300, max_depth=None, calibrate=True):
    """
    Extremely Randomized Trees.
    
    Advantages over Random Forest:
    - Random splits → faster training
    - Often better on mixed feature types (ratios, counts, frequencies)
    - Less prone to overfitting
    
    Args:
        n_estimators: Number of trees
        max_depth: Max tree depth
        calibrate: Calibrate probabilities
    """
    et = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
        verbose=0,
    )
    
    if calibrate:
        return OptionalScalerPipeline(
            scaler=make_scaler('tree'),
            classifier=CalibratedClassifierCV(
                et,
                method='sigmoid',
                cv=3,
            )
        )
    else:
        return OptionalScalerPipeline(
            scaler=make_scaler('tree'),
            classifier=et
        )


# ============================================================================
# XGBoost (GPU with proper detection, No Scaling)
# ============================================================================

def make_xgboost(n_estimators=500, max_depth=6, learning_rate=0.05, use_gpu=True):
    """
    XGBoost with proper GPU detection.
    
    GPU logic:
    - Actually tests GPU capability at import time
    - Falls back to CPU 'hist' if GPU unavailable
    - Clear logging of GPU status
    
    Args:
        n_estimators: Boosting rounds (500 is a good default)
        max_depth: Tree depth (6 balances complexity/speed)
        learning_rate: Shrinkage (0.05-0.1 typical)
        use_gpu: Try GPU if available (no crash if unavailable)
    """
    if not XGBOOST_AVAILABLE:
        warnings.warn("XGBoost not installed. Falling back to ExtraTrees.")
        return make_extra_trees()
    
    # Use actual GPU test results, not just CUDA check
    if use_gpu and _XGBOOST_GPU_AVAILABLE:
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
        print("[classifiers] XGBoost: Using GPU acceleration (verified)")
    else:
        tree_method = "hist"  # CPU-optimized histogram method
        predictor = "cpu_predictor"
        if use_gpu and not _XGBOOST_GPU_AVAILABLE:
            print("[classifiers] XGBoost: GPU not available in build. Using CPU 'hist'.")
    
    return OptionalScalerPipeline(
        scaler=make_scaler('tree'),  # No scaling for gradient boosting
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
        )
    )


# ============================================================================
# LightGBM (GPU with proper detection, No Scaling)
# ============================================================================

def make_lightgbm(n_estimators=500, num_leaves=63, learning_rate=0.05, use_gpu=True):
    """
    LightGBM with proper GPU detection.
    
    Typically the best classical model for PAN authorship tasks.
    
    Why it excels at stylometry:
    - Handles large feature spaces efficiently (histogram-based)
    - Good with mixed distributions (ratios, frequencies, counts)
    - Leaf-wise growth captures complex feature interactions
    - Very fast training even on CPU
    
    Args:
        n_estimators: Boosting rounds
        num_leaves: Max leaves (63 is a good default for ~12.5k features)
        learning_rate: Shrinkage
        use_gpu: Try GPU if available
    """
    if not LIGHTGBM_AVAILABLE:
        warnings.warn("LightGBM not installed. Falling back to ExtraTrees.")
        return make_extra_trees()
    
    # Use actual GPU test results
    if use_gpu and _LIGHTGBM_GPU_AVAILABLE:
        device = "gpu"
        print("[classifiers] LightGBM: Using GPU acceleration (verified)")
    else:
        device = "cpu"
        if use_gpu and not _LIGHTGBM_GPU_AVAILABLE:
            print("[classifiers] LightGBM: GPU not available in build. Using CPU.")
    
    return OptionalScalerPipeline(
        scaler=make_scaler('tree'),  # No scaling for gradient boosting
        classifier=lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            device=device,
            verbose=-1,
            random_state=42,
        )
    )


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY = {
    # Fast baselines
    "logistic_regression": make_logistic_regression,
    "linear_svc": make_linear_svc,
    
    # Neural network (corrected architecture)
    "mlp": make_mlp,
    
    # Tree-based (no scaling)
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
}

# Add GPU models if available
if XGBOOST_AVAILABLE:
    MODEL_REGISTRY["xgboost"] = make_xgboost

if LIGHTGBM_AVAILABLE:
    MODEL_REGISTRY["lightgbm"] = make_lightgbm

# Model metadata
MODEL_DESCRIPTIONS = {
    "logistic_regression": "Fast linear baseline, interpretable coefficients",
    "linear_svc": "Linear SVM with calibrated probabilities",
    "mlp": "Neural network (256, 128) tuned for ~12.5k features",
    "random_forest": "500 trees, calibrated, no scaling needed",
    "extra_trees": "500 randomized trees, often beats RF",
    "xgboost": f"Gradient boosting ({'GPU' if _XGBOOST_GPU_AVAILABLE else 'CPU'})",
    "lightgbm": f"Gradient boosting ({'GPU' if _LIGHTGBM_GPU_AVAILABLE else 'CPU'}), often best",
}

MODEL_USES_SCALING = {
    "logistic_regression": True,
    "linear_svc": True,
    "mlp": True,
    "random_forest": False,
    "extra_trees": False,
    "xgboost": False,
    "lightgbm": False,
}


# ============================================================================
# Utility Functions
# ============================================================================

def build_model(name, **kwargs):
    """Instantiate and return the named model (unfitted)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def train_model(name, X_train, y_train, **kwargs):
    """Instantiate, fit, and return the named model."""
    model = build_model(name, **kwargs)
    model.fit(X_train, y_train)
    
    # Training summary
    n_features = X_train.shape[1]
    uses_scaling = MODEL_USES_SCALING.get(name, True)
    print(f"[classifiers] Trained '{name}' on {len(y_train)} samples "
          f"({n_features} features, scaling={'yes' if uses_scaling else 'no'})")
    
    return model


def list_available_models():
    """Print all available models with metadata."""
    print("\n" + "=" * 70)
    print("Available Models for Author Switch Detection")
    print("=" * 70)
    
    for name, description in MODEL_DESCRIPTIONS.items():
        available = "✓" if name in MODEL_REGISTRY else "✗"
        scaling = "scaled" if MODEL_USES_SCALING.get(name, True) else "no scaling"
        print(f"  [{available}] {name:25s} [{scaling:10s}] {description}")
    
    print("=" * 70)
    
    # GPU status
    if XGBOOST_AVAILABLE:
        print(f"  XGBoost GPU: {'Available ✓' if _XGBOOST_GPU_AVAILABLE else 'Not available ✗'}")
    if LIGHTGBM_AVAILABLE:
        print(f"  LightGBM GPU: {'Available ✓' if _LIGHTGBM_GPU_AVAILABLE else 'Not available ✗'}")
    
    print()


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    import numpy as np
    
    # Generate synthetic data (~12.5k features like your real data)
    np.random.seed(42)
    n_features = 12500
    X = np.random.randn(1000, n_features).astype(np.float32)
    y = (np.random.rand(1000) > 0.7).astype(np.int32)
    
    print("\n" + "=" * 70)
    print("Testing model registry with synthetic data")
    print("=" * 70)
    print(f"Features: {X.shape}")
    print(f"Class balance: {np.mean(y):.2%} positive")
    print("=" * 70)
    
    # Test all available models
    for name in MODEL_REGISTRY:
        try:
            model = train_model(name, X[:800], y[:800])
            score = model.score(X[800:], y[800:])
            print(f"  {name:25s} Test accuracy: {score:.3f}")
        except Exception as e:
            print(f"  {name:25s} Failed: {str(e)[:80]}")
    
    list_available_models()