"""
Feature group ablation studies for author switch detection.
Updated to work with improved pipeline and caching.
"""

import csv
import os
import time
import numpy as np
from evaluation.metrics import cross_validate_model
from features.pipeline import ALL_GROUPS, FeaturePipeline
from features.pairwise import build_pairwise_dataset
from models.classifiers import build_model


def run_leave_one_out(problems, model_name="logistic_regression", n_splits=5, 
                      groups=None, use_cache=True):
    """
    Leave-one-out ablation: remove one feature group at a time, measure F1 drop.
    
    A large drop when removing group X → group X is important.
    
    Args:
        problems: list of problem dicts
        model_name: model to use for ablation
        n_splits: CV folds
        groups: feature groups to test (default: ALL_GROUPS)
        use_cache: use cached features if available
    
    Returns:
        list of dicts sorted by F1 drop (most important group first)
    """
    active_groups = groups or ALL_GROUPS
    
    print("\n" + "="*70)
    print("Leave-One-Out Feature Group Ablation")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Groups: {active_groups}")
    print("="*70)
    
    # Baseline: all groups
    print("\n[ablation] Running baseline (all groups) ...")
    fp_full = FeaturePipeline(groups=active_groups)
    X_full, y_full, _, grp_full = build_pairwise_dataset(
        problems, fp_full, use_cache=use_cache
    )
    
    baseline = cross_validate_model(
        build_model(model_name), X_full, y_full, grp_full, n_splits
    )
    baseline_f1 = baseline["test_f1_mean"]
    baseline_std = baseline["test_f1_std"]
    print(f"  → Baseline F1 = {baseline_f1:.4f} ± {baseline_std:.4f}")
    
    results = [
        {
            "group": "ALL",
            "f1": baseline_f1,
            "f1_std": baseline_std,
            "f1_drop": 0.0,
        }
    ]
    
    # Test removing each group
    for group in active_groups:
        ablated = [g for g in active_groups if g != group]
        print(f"\n[ablation] Removing '{group}' (using {ablated}) ...")
        
        fp = FeaturePipeline(groups=ablated)
        X, y, _, grps = build_pairwise_dataset(
            problems, fp, use_cache=use_cache
        )
        
        res = cross_validate_model(
            build_model(model_name), X, y, grps, n_splits
        )
        
        drop = round(baseline_f1 - res["test_f1_mean"], 4)
        results.append({
            "group": f"ALL - {group}",
            "f1": res["test_f1_mean"],
            "f1_std": res["test_f1_std"],
            "f1_drop": drop,
        })
        print(f"  → F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f}  "
              f"(drop = {drop:+.4f})")
    
    # Sort by importance (largest drop first)
    results.sort(key=lambda r: -r["f1_drop"])
    
    print("\n" + "="*70)
    print("Ablation Results (sorted by importance)")
    print("="*70)
    print(f"  {'Feature Group':<30} {'F1':>8} {'Drop':>8}  Importance")
    print(f"  {'-'*70}")
    for r in results[1:]:  # Skip ALL baseline
        bar = "█" * max(0, int(r["f1_drop"] * 50))
        print(f"  {r['group']:<30} {r['f1']:>8.4f} {r['f1_drop']:>+8.4f}  {bar}")
    print("="*70)
    
    return results


def run_single_group(problems, model_name="logistic_regression", n_splits=5,
                     groups=None, use_cache=True):
    """
    Single-group evaluation: train on each feature group in isolation.
    
    Shows the standalone predictive power of each group.
    
    Args:
        problems: list of problem dicts
        model_name: model to use
        n_splits: CV folds
        groups: feature groups to test
        use_cache: use cached features
    
    Returns:
        list of dicts sorted by F1 (best standalone group first)
    """
    active_groups = groups or ALL_GROUPS
    
    print("\n" + "="*70)
    print("Single-Group Feature Evaluation")
    print("="*70)
    print(f"Model: {model_name}")
    print("="*70)
    
    results = []
    
    for group in active_groups:
        print(f"\n[ablation] Testing '{group}' alone ...")
        
        fp = FeaturePipeline(groups=[group])
        X, y, _, grps = build_pairwise_dataset(
            problems, fp, use_cache=use_cache
        )
        
        res = cross_validate_model(
            build_model(model_name), X, y, grps, n_splits
        )
        
        n_features = X.shape[1]
        results.append({
            "group": group,
            "n_features": n_features,
            "f1": res["test_f1_mean"],
            "f1_std": res["test_f1_std"],
        })
        print(f"  → F1 = {res['test_f1_mean']:.4f} ± {res['test_f1_std']:.4f} "
              f"({n_features} features)")
    
    # Sort by F1
    results.sort(key=lambda r: -r["f1"])
    
    print("\n" + "="*70)
    print("Single-Group Results (sorted by F1)")
    print("="*70)
    print(f"  {'Group':<20} {'Features':>10} {'F1':>10}  Performance")
    print(f"  {'-'*70}")
    for r in results:
        bar = "█" * min(40, int(r["f1"] * 50))
        print(f"  {r['group']:<20} {r['n_features']:>10} {r['f1']:>10.4f}  {bar}")
    print("="*70)
    
    return results


def save_ablation_csv(results, path):
    """Write ablation results to a CSV file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not results:
        print("[ablation] No results to save")
        return
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[ablation] Results saved → {path}")