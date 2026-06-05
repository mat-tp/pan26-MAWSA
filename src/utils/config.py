"""
Central configuration for the Author Switch Detector.

All tuneable knobs live here — import from this module everywhere else.
GPU availability is detected at import time so callers don't need to check.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Directory layout
# ─────────────────────────────────────────────────────────────────────────────

# Root of the repository (two levels above this file: src/utils/ → src/ → root)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_SRC)

RAW_DIR      = os.path.join(_ROOT, "dataset", "mawsa26-pan-zenodo-DATA")
OUTPUTS_DIR  = os.path.join(_ROOT, "dataset", "outputs")
CACHE_DIR    = os.path.join(OUTPUTS_DIR, "cache")
MODELS_DIR   = os.path.join(OUTPUTS_DIR, "models")

# Concrete output paths
MODEL_PATH          = os.path.join(MODELS_DIR,  "model.pkl")
CV_RESULTS_PATH     = os.path.join(OUTPUTS_DIR, "cv_results.json")
EVAL_RESULTS_PATH   = os.path.join(OUTPUTS_DIR, "eval_results.json")
PREDICTIONS_PATH    = os.path.join(OUTPUTS_DIR, "predictions.jsonl")
IMPORTANCE_PATH     = os.path.join(OUTPUTS_DIR, "feature_importance.json")
ABLATION_LOO_PATH   = os.path.join(OUTPUTS_DIR, "ablation_loo.csv")
ABLATION_SGL_PATH   = os.path.join(OUTPUTS_DIR, "ablation_single.csv")

# Ensure output directories exist when this module is first imported
for _d in [OUTPUTS_DIR, CACHE_DIR, MODELS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_cuda() -> bool:
    """Return True if a CUDA-capable GPU is visible to PyTorch or cupy."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[config] CUDA GPU detected via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    try:
        import cupy  # type: ignore
        cupy.cuda.Device(0).compute_capability  # raises if no GPU
        print("[config] CUDA GPU detected via CuPy")
        return True
    except Exception:
        pass
    return False


CUDA_AVAILABLE: bool = _detect_cuda()
if not CUDA_AVAILABLE:
    print("[config] No CUDA GPU detected — running on CPU")

# ─────────────────────────────────────────────────────────────────────────────
# Feature pipeline settings
# ─────────────────────────────────────────────────────────────────────────────

# Which feature groups to include.  Matches ALL_GROUPS order in pipeline.py.
ACTIVE_FEATURE_GROUPS = ["lexical", "punctuation", "function_words", "char_ngrams", "pos", "ngram"]

# Set True to expand function-word features to per-word frequency vector
# (adds ~180 features; useful but increases RAM use)
USE_PER_WORD_FW: bool = False

# Minimum sentences a problem must have to contribute training pairs
MIN_SENTENCES_PER_PROBLEM: int = 3

# How to combine left/right sentence vectors into a pair feature
# Options: "diff" | "concat" | "cosine" | "euclidean" | "combined"
PAIRWISE_MODE: str = "diff"

# ─────────────────────────────────────────────────────────────────────────────
# Model / training settings
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_MODEL: str = "lightgbm"   # best overall for stylometry

# Models run during compare_all_models() cross-validation sweep
MODELS_TO_COMPARE = [
    "logistic_regression",
    "extra_trees",
    "lightgbm",
]

RANDOM_SEED: int = 42

# ─────────────────────────────────────────────────────────────────────────────
# Feature importance settings
# ─────────────────────────────────────────────────────────────────────────────

PERM_IMP_REPEATS: int = 5    # shuffles per feature (higher = more stable)
PERM_IMP_TOP_K:   int = 30   # features to display

# ─────────────────────────────────────────────────────────────────────────────
# Cache settings
# ─────────────────────────────────────────────────────────────────────────────

CACHE_ENABLED: bool = True   # set False to always re-extract features