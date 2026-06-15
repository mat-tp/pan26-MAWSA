"""
Main entry point for the Author Switch Detection system.

Key changes vs previous version
---------------------------------
1. Train / Validation / Test split performed BEFORE any model training.
2. All models are trained FULLY on the training fold.
3. Validation metrics are printed AFTER training for every model.
4. Test-set F1 (PAN macro) is reported as the primary official metric.
5. Cross-validation runs on the training split only (no leakage).
6. Feature names tracked through variance filtering via FeatureSelectorWithNames.
7. Positive importance features saved with correct names.
"""

import argparse
import gc
import os
import random
import sys
import time
import warnings

import numpy as np
from scipy.sparse import hstack as sparse_hstack, issparse, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import dataset_stats, flatten_problems, load_all, load_split
from evaluation.ablation import run_leave_one_out, run_single_group, save_ablation_csv
from evaluation.importance import logistic_coefficients, permutation_importance
from evaluation.metrics import (
    compare_all_models,
    evaluate_by_difficulty,
    evaluate_model,
    error_analysis,
    plot_class_distribution,
    plot_training_log,
)
from features.pipeline import FeaturePipeline, FeatureSelectorWithNames
from features.pairwise import build_pairwise_dataset, expand_meta
from models.classifiers import MODEL_REGISTRY, train_model, build_model, DENSE_ONLY_MODELS
from utils.config import (
    ABLATION_LOO_PATH,
    ABLATION_SGL_PATH,
    ACTIVE_FEATURE_GROUPS,
    CV_RESULTS_PATH,
    EVAL_RESULTS_PATH,
    IMPORTANCE_PATH,
    MIN_SENTENCES_PER_PROBLEM,
    MODEL_PATH,
    MODEL_SELECTION_PATH,
    MODELS_TO_COMPARE,
    PAIRWISE_MODE,
    PERM_IMP_REPEATS,
    PERM_IMP_TOP_K,
    ENABLE_PERM_IMPORTANCE,
    ENABLE_HYPERPARAM_SEARCH,
    HYPERPARAM_SEARCH_CV,
    HYPERPARAM_SEARCH_N_ITER,
    HYPERPARAM_SEARCH_METHOD,
    HYPERPARAM_SEARCH_SCORING,
    PLOT_TRAINING_LOG_PATH,
    PREDICTIONS_PATH,
    PRIMARY_MODEL,
    RAW_DIR,
    RANDOM_SEED,
    USE_EMBEDDINGS,
    USE_PER_WORD_FW,
)
from utils.io import load_model, load_pipeline, save_json, save_model, save_predictions, save_pipeline


# ── PAN@CLEF official metric ──────────────────────────────────────────────────

def pan_f1(y_true, y_pred) -> float:
    """Official TIRA / PAN@CLEF metric: macro-averaged F1 over {0, 1}."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_full_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute all standard + PAN metrics in one call."""
    metrics = {
        "pan_f1":    pan_f1(y_true, y_pred),
        "f1_macro":  float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "n_pos":     int(np.sum(y_true)),
        "n_neg":     int(np.sum(y_true == 0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
        try:
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            pass
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def print_model_metrics(name: str, split_label: str, metrics: dict) -> None:
    """Pretty-print a metrics dict for a model evaluation."""
    w = 58
    print(f"\n  ╔{'═' * w}╗")
    print(f"  ║  {name.upper():<{w-2}}║")
    print(f"  ║  Split: {split_label:<{w-10}}║")
    print(f"  ╠{'═' * w}╣")
    print(f"  ║  PAN F1 (macro, OFFICIAL) : {metrics.get('pan_f1', 0):.4f}{' ← TIRA metric':<{w-35}}║")
    print(f"  ║  F1 macro                 : {metrics.get('f1_macro', 0):.4f}{'':>{w-35}}║")
    print(f"  ║  F1 binary (pos class)    : {metrics.get('f1_binary', 0):.4f}{'':>{w-35}}║")
    print(f"  ║  Accuracy                 : {metrics.get('accuracy', 0):.4f}{'':>{w-35}}║")
    print(f"  ║  Precision (macro)        : {metrics.get('precision', 0):.4f}{'':>{w-35}}║")
    print(f"  ║  Recall    (macro)        : {metrics.get('recall', 0):.4f}{'':>{w-35}}║")
    if "roc_auc" in metrics:
        print(f"  ║  ROC-AUC                  : {metrics['roc_auc']:.4f}{'':>{w-35}}║")
    if "avg_precision" in metrics:
        print(f"  ║  Avg Precision            : {metrics['avg_precision']:.4f}{'':>{w-35}}║")
    if all(k in metrics for k in ("tn", "fp", "fn", "tp")):
        print(f"  ║  Confusion   TN={metrics['tn']:5d}  FP={metrics['fp']:5d}{'':>{w-30}}║")
        print(f"  ║              FN={metrics['fn']:5d}  TP={metrics['tp']:5d}{'':>{w-30}}║")
    print(f"  ║  Samples: {metrics.get('n_samples', '?')} (pos={metrics.get('n_pos', '?')} neg={metrics.get('n_neg', '?')}){'':>{max(0,w-45)}}║")
    print(f"  ╚{'═' * w}╝")


# ── Memory helpers ─────────────────────────────────────────────────────────────

def _ram_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        return float("nan")


def _log_ram(tag: str) -> float:
    mb = _ram_mb()
    if not np.isnan(mb):
        print(f"[RAM] {tag}: {mb:.0f} MB")
    return mb


def _free(*arrays) -> None:
    for a in arrays:
        del a
    gc.collect()


# ── Pipeline factory ───────────────────────────────────────────────────────────

def build_pipeline() -> FeaturePipeline:
    groups = list(ACTIVE_FEATURE_GROUPS)
    if USE_EMBEDDINGS:
        groups.append("embeddings")
    return FeaturePipeline(groups=groups, use_per_word_fw=USE_PER_WORD_FW)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_train_data(data_root: str, subset: float = 1.0) -> list:
    data     = load_all(data_root, splits=("train",))
    problems = flatten_problems(data)
    del data
    gc.collect()

    if subset < 1.0:
        random.seed(RANDOM_SEED)
        k        = max(1, int(len(problems) * subset))
        problems = random.sample(problems, k)
        problems = list(problems)
        gc.collect()

    print(f"\n[main] Total training problems (subset={subset}): {len(problems)}")
    for k, v in dataset_stats(problems).items():
        print(f"  {k}: {v}")
    _log_ram("after load_train_data")
    return problems


# ── Evaluate a single fitted model on a split ──────────────────────────────────

def _eval_model_on_split(model, X_split, y_split, name: str, split_label: str) -> dict:
    """Run prediction and compute all metrics. Returns metrics dict."""
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_split)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_split)
    else:
        y_prob = None
    y_pred = model.predict(X_split)
    metrics = compute_full_metrics(y_split, y_pred, y_prob)
    print_model_metrics(name, split_label, metrics)
    return metrics


# ── Full training + validation + test loop ─────────────────────────────────────

def run_train(
    data_root: str,
    subset: float = 0.1,
    use_hyperparam_search: bool = ENABLE_HYPERPARAM_SEARCH,
    search_method: str = HYPERPARAM_SEARCH_METHOD,
    search_cv: int = HYPERPARAM_SEARCH_CV,
    search_n_iter: int = HYPERPARAM_SEARCH_N_ITER,
    search_scoring: str = HYPERPARAM_SEARCH_SCORING,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Train ALL models, validate and test AFTER training, select best by val F1."""
    t_start      = time.time()
    training_log = []

    def _checkpoint(step: str) -> None:
        training_log.append({
            "step":      step,
            "elapsed_s": round(time.time() - t_start, 1),
            "ram_mb":    _log_ram(step),
        })

    # ── 1. Load & feature extraction ──────────────────────────────────────────
    problems = load_train_data(data_root, subset=subset)
    _checkpoint("load_train_data")

    fp = build_pipeline()
    fp.describe()

    print("\n[main] Fitting feature pipeline on ALL training sentences...")
    all_sentences = [s for p in problems for s in p["sentences"]]
    fp.fit(all_sentences)
    del all_sentences
    gc.collect()
    _checkpoint("fit_pipeline")

    print(f"\n[main] Building pairwise dataset (mode={PAIRWISE_MODE}) ...")
    ds = build_pairwise_dataset(
        problems,
        feature_pipeline=fp,
        min_sentences=MIN_SENTENCES_PER_PROBLEM,
        mode=PAIRWISE_MODE,
        use_cache=False,
        cache_sentence_features=False,
    )
    _checkpoint("build_pairwise_dataset")

    pipeline_path = os.path.join(os.path.dirname(MODEL_PATH), "feature_pipeline.pkl")
    save_pipeline(fp, pipeline_path)
    _checkpoint("save_pipeline")

    _free(problems)
    problems = None

    X_dense_raw, y, problem_meta, groups = ds.to_memory()

    if ds.n_sparse > 0:
        print("[main] Loading sparse char-gram features into memory...")
        X_sparse_raw = ds.X_sparse_unsafe()
        # Convert dense to CSR *before* hstack to avoid the huge COO
        # intermediate that scipy builds when mixing dense+sparse inputs.
        # That intermediate caused an OOM at ~3 GB for 2.4 M pairs x 835 cols.
        if not issparse(X_dense_raw):
            print("[main] Converting dense block to CSR for memory-safe hstack...")
            X_dense_csr = csr_matrix(X_dense_raw.astype(np.float32))
        else:
            X_dense_csr = X_dense_raw
        X = sparse_hstack([X_dense_csr, X_sparse_raw], format="csr")
        del X_dense_csr, X_sparse_raw
    else:
        X = X_dense_raw

    if hasattr(X, "dtype") and X.dtype == np.float64:
        X = X.astype(np.float32)
        print("[main] Downcasted X to float32")
    _checkpoint("to_memory")

    print(f"[main] Feature matrix shape: {X.shape}")
    print(f"[main] Pipeline feature count: {fp.n_features}")
    assert X.shape[1] == fp.n_features, (
        f"Feature mismatch: X has {X.shape[1]}, pipeline has {fp.n_features}"
    )

    # ── 2. Variance thresholding with name tracking ──────────────────────────
    print("\n[main] Applying variance threshold ...")
    var_selector = FeatureSelectorWithNames(threshold=0.01)
    X_filtered = var_selector.fit_transform(X)
    
    # Get original feature names from pipeline
    all_names = fp.get_feature_names()
    
    # Safety: ensure names match feature count
    if len(all_names) != X.shape[1]:
        print(f"[main] WARNING: Name count ({len(all_names)}) != feature count ({X.shape[1]})")
        all_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Track which features survived variance filtering
    support = var_selector.get_support()
    if support is not None:
        survived_feature_names = [name for name, keep in zip(all_names, support) if keep]
    else:
        survived_feature_names = [f"feature_{i}" for i in range(X_filtered.shape[1])]
    
    # LOCK final names into pipeline (single source of truth)
    fp.set_final_feature_names(survived_feature_names)
    
    removed_pct = 100 * (1 - X_filtered.shape[1] / X.shape[1])
    print(f"[main] Variance filter: {X.shape[1]} → {X_filtered.shape[1]} features "
          f"({removed_pct:.1f}% removed)")
    print(f"[main] Survived features sample: {survived_feature_names[:5]}...")
    
    # ── Convert to dense for models that need it ─────────────────────────────
    X_sparse = None
    X_dense = None
    
    if issparse(X_filtered):
        X_sparse = X_filtered
        print("[main] Converting filtered features to dense for NN models...")
        X_dense = X_filtered.toarray()
        print(f"[main] Dense array size: {X_dense.shape}, memory: {X_dense.nbytes / 1024**2:.1f} MB")
    else:
        X_dense = X_filtered
        X_sparse = X_filtered
    
    selector_path = os.path.join(os.path.dirname(MODEL_PATH), "variance_selector.pkl")
    save_model(var_selector, selector_path)
    _checkpoint("variance_threshold")

    print(f"[main] X shape: {X_dense.shape}  |  "
          f"y: {(y==0).sum()} same-author / {(y==1).sum()} switch")

    try:
        plot_class_distribution(y)
    except Exception as e:
        print(f"[warning] Class distribution plot failed: {e}")

    # ── 3. Stratified train / val / test split ────────────────────────────────
    n_total = len(y)
    idx_all = np.arange(n_total)

    idx_train, idx_tmp, y_train, y_tmp = train_test_split(
        idx_all, y,
        test_size=(val_frac + test_frac),
        stratify=y,
        random_state=RANDOM_SEED,
    )
    half_test = test_frac / (val_frac + test_frac)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_tmp, y_tmp,
        test_size=half_test,
        stratify=y_tmp,
        random_state=RANDOM_SEED,
    )

    X_train_dense = X_dense[idx_train]
    X_val_dense   = X_dense[idx_val]
    X_test_dense  = X_dense[idx_test]
    
    X_train_sparse = X_sparse[idx_train] if issparse(X_sparse) else X_train_dense
    X_val_sparse   = X_sparse[idx_val] if issparse(X_sparse) else X_val_dense
    X_test_sparse  = X_sparse[idx_test] if issparse(X_sparse) else X_test_dense

    print(f"\n[main] Splits → "
          f"train={len(y_train)} | val={len(y_val)} | test={len(y_test)}")
    print(f"  Train pos={y_train.sum()} neg={(y_train==0).sum()} "
          f"({100*y_train.mean():.1f}% switch)")
    print(f"  Val   pos={y_val.sum()}   neg={(y_val==0).sum()} "
          f"({100*y_val.mean():.1f}% switch)")
    print(f"  Test  pos={y_test.sum()}  neg={(y_test==0).sum()} "
          f"({100*y_test.mean():.1f}% switch)")

    _checkpoint("train_val_test_split")

    # ── 4. Cross-validate on TRAIN split only ──────────────────────────────────
    print(f"\n[main] Cross-validating {len(MODELS_TO_COMPARE)} model(s) on TRAIN split ...")
    train_groups = groups[idx_train] if groups is not None else None
    cv_results = compare_all_models(X_train_dense, y_train, train_groups)
    save_json(cv_results, CV_RESULTS_PATH)
    _checkpoint("cross_validation")

    # ── 5. Train ALL models fully, then evaluate on val + test ────────────────
    trained_models   = {}
    val_metrics_all  = {}
    test_metrics_all = {}

    print(f"\n{'='*65}")
    print(f"  Training ALL {len(MODELS_TO_COMPARE)} models on FULL train split")
    print(f"{'='*65}")

    for model_name in MODELS_TO_COMPARE:
        if model_name not in MODEL_REGISTRY:
            print(f"[main] Skipping unavailable model: {model_name}")
            continue
        try:
            t0 = time.time()
            print(f"\n  → Training: {model_name.upper()}")
            
            if model_name in DENSE_ONLY_MODELS:
                X_tr = X_train_dense
                X_v  = X_val_dense
                X_te = X_test_dense
            else:
                X_tr = X_train_sparse
                X_v  = X_val_sparse
                X_te = X_test_sparse
            
            model = train_model(
                model_name, X_tr, y_train,
                use_hyperparam_search=use_hyperparam_search,
                search_method=search_method,
                search_cv=search_cv,
                search_n_iter=search_n_iter,
                search_scoring=search_scoring,
            )
            elapsed = time.time() - t0
            print(f"     Training time: {elapsed:.1f}s")

            val_m = _eval_model_on_split(model, X_v, y_val, model_name, "VALIDATION")
            val_metrics_all[model_name] = val_m

            test_m = _eval_model_on_split(model, X_te, y_test, model_name, "TEST (PAN F1 ← official)")
            test_metrics_all[model_name] = test_m

            model_path = os.path.join(os.path.dirname(MODEL_PATH), f"{model_name}.pkl")
            save_model(model, model_path)
            trained_models[model_name] = model

        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {model_name}: {exc}")
            traceback.print_exc()
            val_metrics_all[model_name]  = {"pan_f1": 0.0, "error": str(exc)}
            test_metrics_all[model_name] = {"pan_f1": 0.0, "error": str(exc)}

    _checkpoint("train_all_models")

    # ── 6. Final summary table ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY — ALL MODELS")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'Val F1':>8}  {'Test F1':>8}  {'Test AUC':>9}")
    print(f"  {'-'*58}")

    best_name, best_val_f1 = None, -1.0
    for name in MODELS_TO_COMPARE:
        if name not in val_metrics_all:
            continue
        v_f1 = val_metrics_all[name].get("pan_f1", 0.0)
        t_f1 = test_metrics_all[name].get("pan_f1", 0.0)
        t_auc = test_metrics_all[name].get("roc_auc", float("nan"))
        err = val_metrics_all[name].get("error", "")
        flag = " ← BEST" if v_f1 > best_val_f1 and not err else ""
        if not err:
            print(f"  {name:<25} {v_f1:>8.4f}  {t_f1:>8.4f}  {t_auc:>9.4f}{flag}")
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                best_name   = name
        else:
            print(f"  {name:<25}   ERROR: {err[:40]}")
    print(f"{'='*65}\n")

    # ── 7. Save best model ────────────────────────────────────────────────────
    if best_name is None:
        if trained_models:
            best_name = list(trained_models.keys())[0]
            best_val_f1 = val_metrics_all[best_name].get("pan_f1", 0.0)
        elif PRIMARY_MODEL in MODEL_REGISTRY:
            best_name = PRIMARY_MODEL
            best_val_f1 = 0.0
        else:
            print("[main] No models trained successfully. Exiting.")
            return None, fp, X_dense, y, problem_meta, groups, None
    
    if best_name in trained_models:
        best_model = trained_models[best_name]
        save_model(best_model, MODEL_PATH)
        print(f"[main] Best model saved: {best_name} (Val PAN F1 = {best_val_f1:.4f})")
    else:
        best_model = None
        print(f"[main] Warning: Best model '{best_name}' not found in trained models.")

    _checkpoint("save_best_model")

    save_json({
        "best_model":    best_name,
        "best_val_f1":   best_val_f1,
        "best_test_f1":  test_metrics_all.get(best_name, {}).get("pan_f1", 0.0),
        "feature_count": int(X_dense.shape[1]),
        "original_feature_count": len(all_names),
        "survived_features": survived_feature_names,
        "split": {
            "train": int(len(y_train)),
            "val":   int(len(y_val)),
            "test":  int(len(y_test)),
        },
        "val_metrics":  {
            k: {mk: mv for mk, mv in v.items() if not isinstance(mv, np.ndarray)}
            for k, v in val_metrics_all.items()
        },
        "test_metrics": {
            k: {mk: mv for mk, mv in v.items() if not isinstance(mv, np.ndarray)}
            for k, v in test_metrics_all.items()
        },
        "cv_results":  cv_results,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }, MODEL_SELECTION_PATH)
    print(f"[main] Model selection saved → {MODEL_SELECTION_PATH}")

    # ── 8. Permutation importance ─────────────────────────────────────────────
    if ENABLE_PERM_IMPORTANCE and best_model is not None:
        print("\n[main] Computing permutation importance on best model ...")
        MAX_IMP_ROWS = 5_000
        n_rows = X_train_dense.shape[0]
        
        if n_rows > MAX_IMP_ROWS:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(n_rows, MAX_IMP_ROWS, replace=False)
            X_imp, y_imp = X_train_dense[idx], y_train[idx]
        else:
            X_imp, y_imp = X_train_dense, y_train

        # Use final feature names from pipeline (SINGLE SOURCE OF TRUTH)
        final_names = fp.get_final_feature_names()
        print(f"[main] Using {len(final_names)} final feature names for importance")
        
        imp = permutation_importance(
            best_model,
            X_imp,
            y_imp,
            feature_names=final_names,
            n_repeats=PERM_IMP_REPEATS,
            top_k=PERM_IMP_TOP_K,
        )
        save_json(imp, IMPORTANCE_PATH)
        
        # ── Build positive features list ──────────────────────────────────
        POS_FEATURES_PATH = os.path.join(os.path.dirname(IMPORTANCE_PATH), "positive_features.json")
        
        # Extract importance values
        if isinstance(imp, dict):
            importances_mean = imp.get("importances_mean", [])
            importances_std  = imp.get("importances_std", [])
        elif isinstance(imp, list):
            importances_mean = [item.get("importance", item.get("importance_mean", 0.0)) for item in imp]
            importances_std  = [item.get("std", item.get("importance_std", 0.0)) for item in imp]
        else:
            importances_mean = []
            importances_std  = []
        
        positive_count = sum(1 for v in importances_mean if v > 0)
        
        positive_features = {
            "model": best_name,
            "n_total_features": int(X_dense.shape[1]),
            "n_positive_features": positive_count,
            "positive_features": []
        }
        
        for i, name in enumerate(final_names):
            if i >= len(importances_mean):
                continue
            if importances_mean[i] > 0:
                positive_features["positive_features"].append({
                    "rank": len(positive_features["positive_features"]) + 1,
                    "feature_name": name,
                    "importance_mean": float(importances_mean[i]),
                    "importance_std": float(importances_std[i]),
                })
        
        # Print summary
        print(f"\n[main] Positive importance features: {positive_count}/{len(final_names)}")
        for i, f in enumerate(positive_features["positive_features"][:30], 1):
            print(f"  Rank {i:2d}: {f['feature_name']:<40s} (mean={f['importance_mean']:.4f}, std={f['importance_std']:.4f})")
        
        save_json(positive_features, POS_FEATURES_PATH)
        print(f"\n[main] Positive features saved → {POS_FEATURES_PATH}")
        
        _free(X_imp, y_imp)
        _checkpoint("permutation_importance")
    else:
        print("[main] Permutation importance skipped (ENABLE_PERM_IMPORTANCE=False)")

    if best_name == "logistic_regression" and best_model is not None:
        coef = logistic_coefficients(best_model, feature_names=fp.get_final_feature_names())
        save_json(coef, IMPORTANCE_PATH.replace(".json", "_coeff.json"))

    _checkpoint("done")
    plot_training_log(training_log, path=PLOT_TRAINING_LOG_PATH)

    return best_model, fp, X_dense, y, problem_meta, groups, best_name


# ── Ablation ───────────────────────────────────────────────────────────────────

def run_ablation(data_root: str) -> None:
    problems = load_train_data(data_root)
    print("\n[main] Running leave-one-out ablation ...")
    loo = run_leave_one_out(problems, model_name=PRIMARY_MODEL)
    save_ablation_csv(loo, ABLATION_LOO_PATH)
    gc.collect()
    print("\n[main] Running single-group ablation ...")
    sgl = run_single_group(problems, model_name=PRIMARY_MODEL)
    save_ablation_csv(sgl, ABLATION_SGL_PATH)
    gc.collect()
    print("\n[main] Ablation complete.")


# ── Prediction ─────────────────────────────────────────────────────────────────

def get_pipeline_path(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), "feature_pipeline.pkl")


def get_selector_path(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), "variance_selector.pkl")


def run_predict(data_root: str, model_path: str = None) -> list:
    """Load a trained model and generate predictions for the validation split."""
    model_path = model_path or MODEL_PATH
    model      = load_model(model_path)

    pipeline_path = get_pipeline_path(model_path)
    if os.path.exists(pipeline_path):
        fp = load_pipeline(pipeline_path)
    else:
        raise FileNotFoundError(f"Feature pipeline not found at {pipeline_path}.")

    selector_path = get_selector_path(model_path)
    if os.path.exists(selector_path):
        selector = load_model(selector_path)
    else:
        raise FileNotFoundError(f"Variance selector not found at {selector_path}.")

    data     = load_all(data_root, splits=("validation",))
    problems = flatten_problems(data)
    del data
    gc.collect()

    if not problems:
        problems = load_split(data_root, difficulty="unknown")

    print(f"[main] Predicting on {len(problems)} problems ...")
    predictions = []

    for problem in problems:
        sentences = problem["sentences"]
        if len(sentences) < 2:
            predictions.append({"problem_id": problem["problem_id"], "changes": []})
            continue

        vecs    = fp.extract_batch(sentences)
        vecs    = selector.transform(vecs)
        changes = []
        for i in range(len(sentences) - 1):
            pair_vec = np.abs(vecs[i] - vecs[i + 1]).reshape(1, -1).astype(np.float32)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                changes.append(int(model.predict(pair_vec)[0]))
        predictions.append({"problem_id": problem["problem_id"], "changes": changes})
        del vecs
        gc.collect()

    save_predictions(predictions, PREDICTIONS_PATH)
    print(f"[main] Predictions written → {PREDICTIONS_PATH}")
    return predictions


# ── Full pipeline ──────────────────────────────────────────────────────────────

def run_full(
    data_root: str,
    subset: float = 1.0,
    use_hyperparam_search: bool = ENABLE_HYPERPARAM_SEARCH,
    search_method: str = HYPERPARAM_SEARCH_METHOD,
    search_cv: int = HYPERPARAM_SEARCH_CV,
    search_n_iter: int = HYPERPARAM_SEARCH_N_ITER,
    search_scoring: str = HYPERPARAM_SEARCH_SCORING,
) -> None:
    best_model, fp, X, y, problem_meta, groups, best_model_name = run_train(
        data_root, subset=subset,
        use_hyperparam_search=use_hyperparam_search,
        search_method=search_method, search_cv=search_cv,
        search_n_iter=search_n_iter, search_scoring=search_scoring,
    )
    
    if best_model is None:
        print("[main] No model trained — skipping evaluation.")
        return
    
    _log_ram("run_full: after train")
    print("\n[main] In-sample diagnostic evaluation ...")
    eval_res  = evaluate_model(best_model, X, y, model_name=best_model_name)
    pair_meta = expand_meta(problem_meta, y)
    print("\n[main] Per-difficulty breakdown:")
    diff_res  = evaluate_by_difficulty(best_model, X, y, pair_meta)
    eval_res["by_difficulty"] = diff_res
    print("\n[main] Error analysis:")
    errors = error_analysis(best_model, X, y, pair_meta)
    _free(X, y)
    save_json({
        **eval_res,
        "errors_summary": {
            "false_positives": len(errors["false_positives"]),
            "false_negatives": len(errors["false_negatives"]),
        },
    }, EVAL_RESULTS_PATH)
    run_predict(data_root)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Author Switch Detector")
    parser.add_argument("--mode", choices=["train", "ablation", "predict", "full"], default="full")
    parser.add_argument("--data", default=RAW_DIR)
    parser.add_argument("--model", default=None)
    parser.add_argument("--subset", type=float, default=0.005)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--hyperparam-search", action="store_true")
    parser.add_argument("--search-method", choices=["grid", "randomized"], default=None)
    parser.add_argument("--search-iter", type=int, default=None)
    args = parser.parse_args()

    search_method = args.search_method or HYPERPARAM_SEARCH_METHOD
    search_n_iter = args.search_iter or HYPERPARAM_SEARCH_N_ITER

    if args.mode == "train":
        run_train(args.data, subset=args.subset,
                  use_hyperparam_search=args.hyperparam_search,
                  search_method=search_method, search_cv=HYPERPARAM_SEARCH_CV,
                  search_n_iter=search_n_iter, search_scoring=HYPERPARAM_SEARCH_SCORING,
                  val_frac=args.val_frac, test_frac=args.test_frac)
    elif args.mode == "ablation":
        run_ablation(args.data)
    elif args.mode == "predict":
        run_predict(args.data, model_path=args.model)
    elif args.mode == "full":
        run_full(args.data, subset=args.subset,
                 use_hyperparam_search=args.hyperparam_search,
                 search_method=search_method, search_cv=HYPERPARAM_SEARCH_CV,
                 search_n_iter=search_n_iter, search_scoring=HYPERPARAM_SEARCH_SCORING)


if __name__ == "__main__":
    main()