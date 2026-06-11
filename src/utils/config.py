"""
Central configuration for the Author Switch Detector.

All tuneable knobs live here — import from this module everywhere else.
GPU availability is detected at import time so callers don't need to check.
"""

import os

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

# Root: src/utils/ → src/ → root
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)

RAW_DIR     = os.path.join(_ROOT, "dataset", "mawsa26-pan-zenodo-DATA")
OUTPUTS_DIR = os.path.join(_ROOT, "dataset", "outputs")
CACHE_DIR   = os.path.join(OUTPUTS_DIR, "cache")
MODELS_DIR  = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR   = os.path.join(OUTPUTS_DIR, "plots")

# Output file paths
MODEL_PATH           = os.path.join(MODELS_DIR,  "best_model.pkl")
MODEL_SELECTION_PATH = os.path.join(OUTPUTS_DIR, "model_selection.json")
CV_RESULTS_PATH      = os.path.join(OUTPUTS_DIR, "cv_results.json")
EVAL_RESULTS_PATH    = os.path.join(OUTPUTS_DIR, "eval_results.json")
PREDICTIONS_PATH     = os.path.join(OUTPUTS_DIR, "predictions.jsonl")
IMPORTANCE_PATH      = os.path.join(OUTPUTS_DIR, "feature_importance.json")
ABLATION_LOO_PATH    = os.path.join(OUTPUTS_DIR, "ablation_loo.csv")
ABLATION_SGL_PATH    = os.path.join(OUTPUTS_DIR, "ablation_single.csv")

# Plot output paths (PNG, no plt.show())
PLOT_CV_PATH           = os.path.join(PLOTS_DIR, "cv_model_comparison.png")
PLOT_CONFUSION_PATH    = os.path.join(PLOTS_DIR, "confusion_matrix.png")
PLOT_ROC_PATH          = os.path.join(PLOTS_DIR, "roc_curve.png")
PLOT_PR_PATH           = os.path.join(PLOTS_DIR, "precision_recall_curve.png")
PLOT_THRESHOLD_PATH    = os.path.join(PLOTS_DIR, "threshold_optimization.png")
PLOT_IMPORTANCE_PATH   = os.path.join(PLOTS_DIR, "feature_importance.png")
PLOT_ABLATION_LOO_PATH = os.path.join(PLOTS_DIR, "ablation_loo.png")
PLOT_ABLATION_SGL_PATH = os.path.join(PLOTS_DIR, "ablation_single.png")
PLOT_TRAINING_LOG_PATH = os.path.join(PLOTS_DIR, "training_log.png")
PLOT_CLASS_DIST_PATH   = os.path.join(PLOTS_DIR, "class_distribution.png")

# Create output directories on first import
for _d in [OUTPUTS_DIR, CACHE_DIR, MODELS_DIR, PLOTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def _detect_cuda() -> bool:
    """Return True if a CUDA-capable GPU is visible to PyTorch or CuPy."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[config] CUDA GPU detected via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    try:
        import cupy  # type: ignore
        cupy.cuda.Device(0).compute_capability
        print("[config] CUDA GPU detected via CuPy")
        return True
    except Exception:
        pass
    return False


CUDA_AVAILABLE: bool = _detect_cuda()
if not CUDA_AVAILABLE:
    print("[config] No CUDA GPU detected — running on CPU")

# ---------------------------------------------------------------------------
# Feature pipeline settings
# ---------------------------------------------------------------------------

# Active feature groups (matches ALL_GROUPS order in pipeline.py — do not reorder)
# "char_ngrams" & "ngram" are not used by default
ACTIVE_FEATURE_GROUPS = [
    "lexical",
    "punctuation",
    "function_words",
    "readability",
    "char_ngrams",
    "pos",
    "ngram",
]

# Optional embedding feature group. Set to True only if sentence-transformers is installed
# and you want to compare embedding-based representations.
USE_EMBEDDINGS: bool = False
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# Expand function-word features to per-word frequency vector (~180 extra features, more RAM)
USE_PER_WORD_FW: bool = False

# Minimum sentences per problem to generate training pairs
MIN_SENTENCES_PER_PROBLEM: int = 3

# How to combine left/right sentence vectors into a pair feature
# Options: "diff" | "concat" | "cosine" | "euclidean" | "combined"
PAIRWISE_MODE: str = "diff"

# ---------------------------------------------------------------------------
# Model / training settings
# ---------------------------------------------------------------------------

AUTO_SELECT_BEST_MODEL = True
PRIMARY_MODEL = "lightgbm"  # best overall for stylometry

# Models evaluated during compare_all_models() cross-validation sweep
MODELS_TO_COMPARE = [
    "logistic_regression",
    "linear_svc",
    "mlp",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
]

RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Feature importance settings
# ---------------------------------------------------------------------------

PERM_IMP_REPEATS: int = 5   # shuffles per feature (higher = more stable estimates)
PERM_IMP_TOP_K:   int = 30  # number of features to display
ENABLE_PERM_IMPORTANCE: bool = False  # set False to skip permutation importance entirely

# Hyperparameter search settings
ENABLE_HYPERPARAM_SEARCH: bool = False
HYPERPARAM_SEARCH_METHOD: str = "randomized"  # 'grid' or 'randomized'
HYPERPARAM_SEARCH_N_ITER: int = 25
HYPERPARAM_SEARCH_CV: int = 5
HYPERPARAM_SEARCH_SCORING: str = "f1"

# ---------------------------------------------------------------------------
# Cache settings
# ---------------------------------------------------------------------------

CACHE_ENABLED: bool = False  # set False to always re-extract features

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------

PLOT_DPI: int = 150
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"
GENERATE_PLOTS: bool = True  # set False for headless/CI runs