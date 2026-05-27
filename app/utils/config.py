"""
Central configuration for all experiments.

Keeping all settings here means you can reproduce any experiment by
checking this file, and changing one value updates every script that
imports it.

For hyperparameter sweeps, override specific values at the call site rather
than modifying this file — that way the defaults stay clean.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
RAW_DIR     = os.path.join(DATA_DIR, "mawsa26-pan-zenodo-DATA")
OUTPUT_DIR  = os.path.join(DATA_DIR, "outputs")
MODEL_DIR   = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DIFFICULTIES = ("easy", "medium", "hard")
SPLITS       = ("train", "test")
MIN_SENTENCES_PER_PROBLEM = 3   # skip shorter problems

# ---------------------------------------------------------------------------
# Feature pipeline
# ---------------------------------------------------------------------------

# Toggle feature groups on/off here for quick ablation experiments.
# Set to None to use all groups (default).
ACTIVE_FEATURE_GROUPS = None   # None = ALL_GROUPS from pipeline.py

# Use per-function-word frequency features (adds ~150 features)
USE_PER_WORD_FW = False

# Pairwise representation mode: "diff" (recommended) or "concat"
PAIRWISE_MODE = "diff"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

RANDOM_SEED   = 42
CV_SPLITS     = 5
PRIMARY_MODEL = "mlp"   # model used for final evaluation and TIRA submission

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Which models to include in the comparison table
MODELS_TO_COMPARE = ["logistic_regression", "svm", "mlp", "naive_bayes"]

# Permutation importance: number of shuffle repeats
PERM_IMP_REPEATS = 10
PERM_IMP_TOP_K   = 25

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------

MODEL_PATH       = os.path.join(MODEL_DIR,   f"{PRIMARY_MODEL}.pkl")
CV_RESULTS_PATH  = os.path.join(RESULTS_DIR, "cv_comparison.json")
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation.json")
ABLATION_LOO_PATH = os.path.join(RESULTS_DIR, "ablation_leave_one_out.csv")
ABLATION_SGL_PATH = os.path.join(RESULTS_DIR, "ablation_single_group.csv")
IMPORTANCE_PATH  = os.path.join(RESULTS_DIR, "feature_importance.json")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR,  "predictions.jsonl")
