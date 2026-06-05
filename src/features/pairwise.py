"""
Pairwise dataset construction for author switch detection.
Optimized with vectorized pair generation, multi-level caching,
multiple distance metrics, and optional multiprocessing.

Key improvements:
  - Vectorized pair creation: left=feats[:-1], right=feats[1:]
  - Sentence feature cache (reuse across experiments)
  - Pairwise feature cache (reuse across models)
  - Multiple distance modes: diff, concat, cosine, euclidean, combined
  - Optional multiprocessing for sentence extraction
"""

import hashlib
import json
import os
import pickle
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils.config import CACHE_DIR, CACHE_ENABLED
from features.ngram_features import NGramExtractor
from features.pipeline import FeaturePipeline


# ============================================================================
# Cache Management Utilities
# ============================================================================

def _compute_cache_key(problems_hash, feat_config_dict):
    """
    Create a deterministic hash of the input data and feature config.
    
    Args:
        problems_hash: Hash of the problems list content
        feat_config_dict: Feature configuration dictionary
    """
    data_str = json.dumps({
        "problems_hash": problems_hash,
        "feat_config": feat_config_dict,
    }, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def _hash_problems(problems):
    """Create a hash of the problems list for cache identification."""
    prob_info = []
    for p in problems:
        prob_info.append({
            "problem_id": p["problem_id"],
            "difficulty": p["difficulty"],
            "n_sentences": len(p["sentences"]),
            "n_changes": len(p["changes"]),
            "changes_sum": sum(p["changes"]),
        })
    return hashlib.sha256(json.dumps(prob_info, sort_keys=True).encode()).hexdigest()


def _get_pairwise_cache_path(cache_key, mode):
    """Get cache paths for pairwise data and metadata."""
    cache_dir = os.path.join(CACHE_DIR, "pairwise")
    os.makedirs(cache_dir, exist_ok=True)
    
    features_path = os.path.join(cache_dir, f"{mode}_{cache_key}.npz")
    meta_path = os.path.join(cache_dir, f"{mode}_{cache_key}.pkl")
    
    return features_path, meta_path


# ============================================================================
# Distance / Similarity Metrics
# ============================================================================

def compute_pairwise_features(left_features, right_features, mode="diff"):
    """
    Compute pairwise features from left and right sentence feature matrices.
    Fully vectorized - no Python loops.
    
    Args:
        left_features: numpy array (n_pairs, n_features)
        right_features: numpy array (n_pairs, n_features)
        mode: distance mode ('diff', 'concat', 'cosine', 'euclidean', 'combined')
    
    Returns:
        numpy array of pairwise features
    
    Modes:
        diff:      |f1 - f2|                    (n_features)
        concat:    [f1, f2]                      (2 * n_features)
        cosine:    1 - cosine_similarity         (1 feature)
        euclidean: ||f1 - f2||_2                 (1 feature)
        combined:  [diff, cosine, euclidean]     (n_features + 2)
    """
    if mode == "diff":
        return np.abs(left_features - right_features)
    
    elif mode == "concat":
        return np.hstack([left_features, right_features])
    
    elif mode == "cosine":
        # Vectorized cosine similarity
        dot_product = np.sum(left_features * right_features, axis=1)
        norm_left = np.linalg.norm(left_features, axis=1)
        norm_right = np.linalg.norm(right_features, axis=1)
        
        # Avoid division by zero
        denominator = norm_left * norm_right
        cosine_sim = np.where(denominator > 0, dot_product / denominator, 0.0)
        
        # Return distance (1 - similarity) as a column vector
        return (1.0 - cosine_sim).reshape(-1, 1).astype(np.float32)
    
    elif mode == "euclidean":
        # Vectorized Euclidean distance
        diff = left_features - right_features
        euclidean_dist = np.sqrt(np.sum(diff * diff, axis=1))
        return euclidean_dist.reshape(-1, 1).astype(np.float32)
    
    elif mode == "combined":
        # All three metrics combined
        diff_features = np.abs(left_features - right_features)
        
        # Cosine
        dot_product = np.sum(left_features * right_features, axis=1)
        norm_left = np.linalg.norm(left_features, axis=1)
        norm_right = np.linalg.norm(right_features, axis=1)
        denominator = norm_left * norm_right
        cosine_sim = np.where(denominator > 0, dot_product / denominator, 0.0)
        cosine_dist = (1.0 - cosine_sim).reshape(-1, 1)
        
        # Euclidean
        diff_sq = (left_features - right_features) ** 2
        euclidean_dist = np.sqrt(np.sum(diff_sq, axis=1)).reshape(-1, 1)
        
        return np.hstack([diff_features, cosine_dist, euclidean_dist]).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'diff', 'concat', 'cosine', 'euclidean', or 'combined'")


# ============================================================================
# Sentence Feature Extraction (with optional multiprocessing)
# ============================================================================

def _extract_problem_features(problem, feature_pipeline, min_sentences):
    """
    Extract features for a single problem (used in multiprocessing).
    
    Args:
        problem: problem dict
        feature_pipeline: FeaturePipeline instance
        min_sentences: minimum sentences required
        
    Returns:
        tuple of (problem_id, sentences, changes, features, difficulty) or None
    """
    sentences = problem["sentences"]
    changes = problem["changes"]
    
    if len(sentences) < min_sentences:
        return None
    
    try:
        # Extract sentence features
        sent_features = feature_pipeline.extract_batch(sentences)
        return (problem["problem_id"], sentences, changes, sent_features, problem["difficulty"])
    except Exception as e:
        print(f"[pairwise] Error extracting features for problem {problem['problem_id']}: {e}")
        return None


def _extract_sentence_features_sequential(problems, feature_pipeline, min_sentences, use_cache, cache_ids):
    """
    Extract sentence features sequentially.
    
    Returns:
        list of (problem_id, sentences, changes, sent_features, difficulty) tuples
    """
    results = []
    
    for prob_idx, problem in enumerate(problems):
        problem_id = problem["problem_id"]
        difficulty  = problem["difficulty"]
        cache_key   = f"{difficulty}_{problem_id}"
        sentences = problem["sentences"]
        changes = problem["changes"]
        
        if len(sentences) < min_sentences:
            continue
        
        # Try cache first
        if use_cache and cache_ids is not None:
            cached = cache_ids.get(cache_key)
            if cached is not None:
                results.append((problem_id, sentences, changes, cached, difficulty))
                continue
        
        # Extract features
        try:
            sent_features = feature_pipeline.extract_batch(sentences)
            results.append((problem_id, sentences, changes, sent_features, problem["difficulty"]))
            
            # Save to cache
            if use_cache and cache_ids is not None:
                cache_ids[cache_key] = sent_features
        except Exception as e:
            print(f"[pairwise] Error extracting features for problem {problem_id}: {e}")
        
        # Progress indicator
        if (prob_idx + 1) % 50 == 0:
            print(f"[pairwise] Extracted features for {prob_idx + 1}/{len(problems)} problems")
    
    return results


def _extract_sentence_features_parallel(problems, feature_pipeline, min_sentences, n_jobs):
    """
    Extract sentence features using multiprocessing.
    
    Args:
        problems: list of problem dicts
        feature_pipeline: FeaturePipeline instance (will be copied per worker)
        min_sentences: minimum sentences required
        n_jobs: number of parallel workers
        
    Returns:
        list of (problem_id, sentences, changes, sent_features, difficulty) tuples
    """
    extract_func = partial(_extract_problem_features, 
                          feature_pipeline=feature_pipeline,
                          min_sentences=min_sentences)
    
    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(extract_func, p): p["problem_id"] for p in problems}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            
            if len(results) % 50 == 0:
                print(f"[pairwise] Extracted features for {len(results)} problems (parallel)")
    
    return results


# ============================================================================
# Main Pairwise Dataset Builders
# ============================================================================

def build_pairwise_dataset(problems, feature_pipeline=None, min_sentences=3, 
                          mode="diff", use_cache=True, n_jobs=1,
                          cache_sentence_features=True):
    """
    Build pairwise dataset from problems list with full optimizations.
    
    Args:
        problems: list of problem dicts with 'sentences' and 'changes' keys
        feature_pipeline: FeaturePipeline instance (created if None)
        min_sentences: minimum sentences required per problem
        mode: distance mode ('diff', 'concat', 'cosine', 'euclidean', 'combined')
        use_cache: whether to use cached features (pairwise level)
        n_jobs: number of parallel workers for sentence extraction (1 = sequential)
        cache_sentence_features: whether to cache sentence features separately
    
    Returns:
        X: feature matrix (n_samples, n_features)
        y: labels array (n_samples,)
        meta: metadata list for each pair
        groups: group IDs (problem indices) for CV stratification
    
    Example:
        >>> problems = load_all("data/train")
        >>> # Fast with diff mode
        >>> X, y, meta, groups = build_pairwise_dataset(problems)
        >>> # With cosine similarity added
        >>> X, y, meta, groups = build_pairwise_dataset(problems, mode="combined")
    """
    start_time = time.time()
    
    if feature_pipeline is None:
        feature_pipeline = FeaturePipeline()
    
    # ---- Check pairwise cache ----
    if use_cache and CACHE_ENABLED:
        problems_hash = _hash_problems(problems)
        feat_config = {
            "groups": feature_pipeline.groups,
            "use_per_word_fw": feature_pipeline.use_per_word_fw,
            "mode": mode,
            "min_sentences": min_sentences,
            "ngram_enabled": feature_pipeline.ngram_enabled,
            "ngram_max_n": getattr(feature_pipeline._ngram_extractor, 'max_n', None) if feature_pipeline.ngram_enabled else None,
        }
        cache_key = _compute_cache_key(problems_hash, feat_config)
        features_path, meta_path = _get_pairwise_cache_path(cache_key, mode)
        
        if os.path.exists(features_path) and os.path.exists(meta_path):
            print(f"[pairwise] Loading from pairwise cache: {features_path}")
            data = np.load(features_path)
            X = data["X"]
            y = data["y"]
            with open(meta_path, "rb") as f:
                meta, groups = pickle.load(f)
            
            elapsed = time.time() - start_time
            print(f"[pairwise] Loaded from cache in {elapsed:.1f}s: {X.shape[0]} pairs, {X.shape[1]} features")
            return X, y, meta, groups
    
    print("[pairwise] Building pairwise dataset from scratch...")
    
    # ---- Fit n-gram models if needed ----
    if feature_pipeline.ngram_enabled:
        print("[pairwise] Fitting n-gram models...")
        all_sentences = []
        for problem in problems:
            if len(problem["sentences"]) >= min_sentences:
                all_sentences.extend(problem["sentences"])
        feature_pipeline.fit(all_sentences)
        print(f"[pairwise] Fitted on {len(all_sentences)} sentences")
    
    # ---- Load or extract sentence features ----
    # Set up sentence cache
    sentence_cache = {}
    if use_cache and CACHE_ENABLED and cache_sentence_features:
        # Check if sentence features are cached
        problems_hash = _hash_problems(problems)
        sent_cache_dir = os.path.join(CACHE_DIR, "sentence_features", problems_hash[:16])
        
        if os.path.exists(sent_cache_dir):
            print(f"[pairwise] Loading sentence features from cache: {sent_cache_dir}")
            for prob in problems:
                prob_id    = prob["problem_id"]
                diff       = prob["difficulty"]
                ck         = f"{diff}_{prob_id}"
                cache_file = os.path.join(sent_cache_dir, f"{diff}_{prob_id}.npy")
                if os.path.exists(cache_file):
                    sentence_cache[ck] = np.load(cache_file)
            print(f"[pairwise] Loaded {len(sentence_cache)} cached sentence feature matrices")
    
    # Extract sentence features
    print("[pairwise] Extracting sentence features...")
    
    if n_jobs > 1 and len(problems) > 10:
        # Parallel extraction
        print(f"[pairwise] Using {n_jobs} parallel workers")
        extracted = _extract_sentence_features_parallel(
            problems, feature_pipeline, min_sentences, n_jobs
        )
    else:
        # Sequential extraction
        extracted = _extract_sentence_features_sequential(
            problems, feature_pipeline, min_sentences, use_cache, sentence_cache
        )
    
    # Save sentence features to cache
    if use_cache and CACHE_ENABLED and cache_sentence_features and len(sentence_cache) > 0:
        problems_hash = _hash_problems(problems)
        sent_cache_dir = os.path.join(CACHE_DIR, "sentence_features", problems_hash[:16])
        os.makedirs(sent_cache_dir, exist_ok=True)
        
        for ck, feats in sentence_cache.items():
            cache_file = os.path.join(sent_cache_dir, f"{ck}.npy")
            if not os.path.exists(cache_file):
                np.save(cache_file, feats)
        
        print(f"[pairwise] Saved {len(sentence_cache)} sentence feature matrices to cache")
    
    # ---- Build pairwise features (fully vectorized) ----
    print(f"[pairwise] Building pairwise features with mode='{mode}'...")
    
    X_list = []
    y_list = []
    meta_list = []
    groups_list = []
    skipped_problems = 0
    
    for prob_idx, (problem_id, sentences, changes, sent_features, difficulty) in enumerate(extracted):
        n_pairs = len(sentences) - 1
        
        if n_pairs < 1:
            skipped_problems += 1
            continue
        
        # Vectorized pair creation - key optimization!
        left_features = sent_features[:-1]   # (n_pairs, n_features)
        right_features = sent_features[1:]   # (n_pairs, n_features)
        
        # Compute pairwise features (single vectorized operation)
        pair_matrix = compute_pairwise_features(left_features, right_features, mode)
        
        X_list.append(pair_matrix)
        y_list.append(np.array(changes, dtype=np.int32))
        
        # Create metadata for each pair
        for i in range(n_pairs):
            meta_list.append({
                "problem_id": problem_id,
                "difficulty": difficulty,
                "pair_idx": i,
                "sentence1": sentences[i][:100],
                "sentence2": sentences[i+1][:100],
                "is_switch": bool(changes[i]),
            })
            groups_list.append(prob_idx)
        
        # Progress indicator
        if (prob_idx + 1) % 50 == 0:
            print(f"[pairwise] Processed {prob_idx + 1}/{len(extracted)} problems")
    
    if skipped_problems > 0:
        print(f"[pairwise] Skipped {skipped_problems} problems with insufficient pairs")
    
    # ---- Combine all pairs efficiently ----
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list) if y_list else np.array([], dtype=np.int32)
    
    # Ensure contiguous memory layout for GPU compatibility
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)
    
    elapsed = time.time() - start_time
    print(f"[pairwise] Dataset built in {elapsed:.1f}s: {X.shape[0]} pairs, {X.shape[1]} features")
    print(f"[pairwise] Class distribution: {np.sum(y == 0)} same, {np.sum(y == 1)} switch")
    print(f"[pairwise] Memory usage: {X.nbytes / 1024**2:.1f} MB")
    
    # ---- Save to pairwise cache ----
    if use_cache and CACHE_ENABLED:
        problems_hash = _hash_problems(problems)
        feat_config = {
            "groups": feature_pipeline.groups,
            "use_per_word_fw": feature_pipeline.use_per_word_fw,
            "mode": mode,
            "min_sentences": min_sentences,
            "ngram_enabled": feature_pipeline.ngram_enabled,
            "ngram_max_n": getattr(feature_pipeline._ngram_extractor, 'max_n', None) if feature_pipeline.ngram_enabled else None,
        }
        cache_key = _compute_cache_key(problems_hash, feat_config)
        features_path, meta_path = _get_pairwise_cache_path(cache_key, mode)
        
        print(f"[pairwise] Saving to pairwise cache: {features_path}")
        np.savez_compressed(features_path, X=X, y=y)
        with open(meta_path, "wb") as f:
            pickle.dump((meta_list, groups_list), f)
    
    return X, y, meta_list, groups_list


def build_pairwise_from_texts(texts1, texts2, labels=None, feature_pipeline=None, mode="diff"):
    """
    Build pairwise dataset directly from two lists of texts.
    Vectorized for efficiency.
    
    Args:
        texts1: list of first sentences in each pair
        texts2: list of second sentences in each pair
        labels: optional list of labels (0/1)
        feature_pipeline: FeaturePipeline instance
        mode: distance mode ('diff', 'concat', 'cosine', 'euclidean', 'combined')
    
    Returns:
        X: feature matrix
        y: labels (or None if not provided)
        meta: metadata
    """
    if feature_pipeline is None:
        feature_pipeline = FeaturePipeline()
    
    n_pairs = len(texts1)
    
    # Fit n-gram models if needed
    if feature_pipeline.ngram_enabled:
        all_sentences = texts1 + texts2
        feature_pipeline.fit(all_sentences)
    
    # Extract features for all sentences in one batch each (vectorized)
    features1 = feature_pipeline.extract_batch(texts1)
    features2 = feature_pipeline.extract_batch(texts2)
    
    # Compute pairwise features (vectorized)
    X = compute_pairwise_features(features1, features2, mode)
    
    # Create metadata
    meta_list = []
    for i in range(n_pairs):
        meta_list.append({
            "pair_idx": i,
            "sentence1": texts1[i][:100],
            "sentence2": texts2[i][:100],
        })
    
    y = np.array(labels, dtype=np.int32) if labels is not None else None
    
    return X, y, meta_list


# ============================================================================
# Utility Functions
# ============================================================================

def get_pairwise_feature_names(feature_pipeline, mode="diff"):
    """
    Get descriptive names for pairwise features based on mode.
    
    Args:
        feature_pipeline: FeaturePipeline instance
        mode: distance mode used
        
    Returns:
        list of feature name strings
    """
    base_names = feature_pipeline.get_feature_names()
    
    if mode == "diff":
        return [f"diff_{name}" for name in base_names]
    
    elif mode == "concat":
        return [f"left_{name}" for name in base_names] + [f"right_{name}" for name in base_names]
    
    elif mode == "cosine":
        return ["cosine_distance"]
    
    elif mode == "euclidean":
        return ["euclidean_distance"]
    
    elif mode == "combined":
        diff_names = [f"diff_{name}" for name in base_names]
        return diff_names + ["cosine_distance", "euclidean_distance"]
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def clear_pairwise_cache(mode=None):
    """
    Clear pairwise feature cache.
    
    Args:
        mode: specific mode to clear (None = all modes)
    """
    cache_dir = os.path.join(CACHE_DIR, "pairwise")
    
    if not os.path.exists(cache_dir):
        print("[cache] No pairwise cache found")
        return
    
    if mode:
        pattern = f"{mode}_*"
        import glob
        files = glob.glob(os.path.join(cache_dir, pattern))
        for f in files:
            os.remove(f)
        print(f"[cache] Cleared {len(files)} cached files for mode '{mode}'")
    else:
        import shutil
        shutil.rmtree(cache_dir)
        print("[cache] Cleared all pairwise cache")