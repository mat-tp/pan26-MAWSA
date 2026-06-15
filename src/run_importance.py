"""
run_importance.py
-----------------
Run feature importance on already-trained models without re-training.

Place this file inside src/ (next to main.py) and run from the project root:

    python -m src.run_importance

All paths are resolved the same way as main.py and utils/config.py.
If a required file is missing the script tells you exactly what is wrong
and, for the data folder, asks you to provide a path.
"""

import os
import sys
import pickle
import numpy as np

# ── Ensure src/ is on the path (same trick as running via -m src.run_importance)
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── All paths come from the same config as main.py ───────────────────────────
from utils.config import (
    RAW_DIR,
    MODEL_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    IMPORTANCE_PATH,
    RANDOM_SEED,
    PERM_IMP_REPEATS,
    PERM_IMP_TOP_K,
    MIN_SENTENCES_PER_PROBLEM,
    PAIRWISE_MODE,
)
from utils.io import load_model, load_pipeline, save_json
from evaluation.importance import permutation_importance
from data.loader import load_all, flatten_problems   # same as main.py line 35 & 175-176
from features.pairwise import build_pairwise_dataset

PIPELINE_PATH = os.path.join(MODELS_DIR, "feature_pipeline.pkl")
SELECTOR_PATH = os.path.join(MODELS_DIR, "variance_selector.pkl")
POS_FEAT_PATH = os.path.join(OUTPUTS_DIR, "positive_features.json")
MAX_IMP_ROWS  = 5_000   # subsample for speed (matches main.py)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_file(path: str, label: str):
    """Abort with a clear message if a required file is missing."""
    if not os.path.exists(path):
        print(f"\n[ERROR] {label} not found:\n  {path}")
        print("  Make sure you have run training (python -m src.main --mode train) first.")
        sys.exit(1)


def _resolve_data_dir() -> str:
    """
    Return the data directory, using the same default as main.py (RAW_DIR).
    If that does not exist, ask the user for the path.
    """
    if os.path.exists(RAW_DIR):
        return RAW_DIR

    print(f"\n[warning] Default data directory not found:\n  {RAW_DIR}")
    while True:
        path = input("  Please enter the path to your training data folder: ").strip()
        if os.path.exists(path):
            return path
        print(f"  Path does not exist: {path}  — please try again.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Verify all required saved artefacts exist
    _require_file(MODEL_PATH,    "Best model (best_model.pkl)")
    _require_file(PIPELINE_PATH, "Feature pipeline (feature_pipeline.pkl)")
    _require_file(SELECTOR_PATH, "Variance selector (variance_selector.pkl)")

    # 2. Load artefacts
    print(f"\n[importance] Loading best model      : {MODEL_PATH}")
    best_model = load_model(MODEL_PATH)
    print(f"             Model type              : {type(best_model).__name__}")

    print(f"[importance] Loading feature pipeline: {PIPELINE_PATH}")
    fp = load_pipeline(PIPELINE_PATH)

    print(f"[importance] Loading variance selector: {SELECTOR_PATH}")
    with open(SELECTOR_PATH, "rb") as fh:
        var_selector = pickle.load(fh)

    # 3. Final feature names (single source of truth, same as main.py)
    final_names = fp.get_final_feature_names()
    print(f"[importance] Final feature count: {len(final_names)}")
    print(f"             Sample names: {final_names[:5]} ...")

    # 4. Resolve data directory (same default as main.py --data argument)
    data_dir = _resolve_data_dir()
    print(f"\n[importance] Loading data from: {data_dir}")

    # load_all returns a nested dict of splits — flatten exactly like main.py does
    data     = load_all(data_dir, splits=("train",))
    problems = flatten_problems(data)
    print(f"[importance] Loaded {len(problems):,} problems")

    # 5. Build pairwise dataset (same settings as main.py)
    print("[importance] Building pairwise feature matrix ...")
    ds = build_pairwise_dataset(
        problems,
        feature_pipeline=fp,
        min_sentences=MIN_SENTENCES_PER_PROBLEM,
        mode=PAIRWISE_MODE,
        use_cache=False,
        cache_sentence_features=False,
    )

    X_dense, y, _, _ = ds.to_memory()

    # 6. Apply variance selector to get the same feature matrix the model saw
    X_filtered = var_selector.transform(X_dense)
    if hasattr(X_filtered, "toarray"):
        X_filtered = X_filtered.toarray()
    X_filtered = X_filtered.astype(np.float32)

    del X_dense  # free memory

    # 7. Subsample for speed (same cap as main.py)
    n_rows = X_filtered.shape[0]
    if n_rows > MAX_IMP_ROWS:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n_rows, MAX_IMP_ROWS, replace=False)
        X_imp, y_imp = X_filtered[idx], y[idx]
        print(f"[importance] Subsampled {MAX_IMP_ROWS:,} / {n_rows:,} rows")
    else:
        X_imp, y_imp = X_filtered, y

    # 8. Run permutation importance
    #    n_jobs=1 avoids joblib's _posixsubprocess (Unix-only, absent on Python 3.14 Windows)
    print(f"\n[importance] Running permutation importance ({PERM_IMP_REPEATS} repeats, n_jobs=1) ...")
    imp = permutation_importance(
        best_model,
        X_imp,
        y_imp,
        feature_names=final_names,
        n_repeats=PERM_IMP_REPEATS,
        top_k=PERM_IMP_TOP_K,
    )
    save_json(imp, IMPORTANCE_PATH)
    print(f"[importance] Saved → {IMPORTANCE_PATH}")

    # 9. Build positive-features summary (same logic as main.py)
    if isinstance(imp, list):
        means = [x.get("importance", x.get("importance_mean", 0.0)) for x in imp]
        stds  = [x.get("std",        x.get("importance_std",  0.0)) for x in imp]
    elif isinstance(imp, dict):
        means = imp.get("importances_mean", [])
        stds  = imp.get("importances_std",  [])
    else:
        means = stds = []

    positive_features = {
        "n_total_features":    len(final_names),
        "n_positive_features": sum(1 for v in means if v > 0),
        "positive_features":   [],
    }
    rank = 1
    for name, mean, std in zip(final_names, means, stds):
        if mean > 0:
            positive_features["positive_features"].append({
                "rank":             rank,
                "feature_name":     name,
                "importance_mean":  float(mean),
                "importance_std":   float(std),
            })
            rank += 1

    n_pos = positive_features["n_positive_features"]
    print(f"\n[importance] Positive features: {n_pos} / {len(final_names)}")
    for f in positive_features["positive_features"][:30]:
        print(f"  Rank {f['rank']:2d}: {f['feature_name']:<40s}  "
              f"mean={f['importance_mean']:.4f}  std={f['importance_std']:.4f}")

    save_json(positive_features, POS_FEAT_PATH)
    print(f"\n[importance] Positive features saved → {POS_FEAT_PATH}")


if __name__ == "__main__":
    main()