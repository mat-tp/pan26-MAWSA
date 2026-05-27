"""
Pairwise sentence representation for author switch detection.

The core idea: given two consecutive sentences (S_i, S_{i+1}), we compute
the absolute difference of their feature vectors:

    x_i = |f(S_i) - f(S_{i+1})|

This representation is:
  - Symmetric: the pair order doesn't matter (important for consistency)
  - Interpretable: large values in x_i mean the two sentences differ a lot
    on that feature, which is evidence of an author switch
  - Compact: the pairwise vector has the same size as a single sentence vector

An alternative is concatenation [f(S_i) || f(S_{i+1})], which the pipeline
also supports via mode="concat" for comparison.
"""

import numpy as np


def build_pairwise_dataset(problems, feature_pipeline, min_sentences=3,
                           mode="diff"):
    """
    Build the training matrix X, label vector y, and metadata.

    Parameters
    ----------
    problems : list of problem dicts (from loader.py)
    feature_pipeline : FeaturePipeline instance
    min_sentences : int
        Skip problems with fewer than this many sentences.
    mode : "diff" | "concat"
        "diff"   → x_i = |f(S_i) - f(S_{i+1})|  (default, recommended)
        "concat" → x_i = [f(S_i), f(S_{i+1})]

    Returns
    -------
    X : ndarray, shape (n_pairs, n_features)
    y : ndarray, shape (n_pairs,), dtype int32
    meta : list of dicts with problem_id, difficulty, pair_index
    groups : ndarray, shape (n_pairs,) — problem index for GroupKFold
    """
    X_rows, y_rows, meta, groups = [], [], [], []
    problem_idx = 0

    for problem in problems:
        sentences = problem["sentences"]
        changes   = problem["changes"]

        if len(sentences) < min_sentences:
            continue

        # Extract features for every sentence in the document.
        vecs = feature_pipeline.extract_document(sentences)

        for i, label in enumerate(changes):
            if mode == "diff":
                pair_vec = np.abs(vecs[i] - vecs[i + 1])
            elif mode == "concat":
                pair_vec = np.concatenate([vecs[i], vecs[i + 1]])
            else:
                raise ValueError(f"Unknown mode: {mode!r}. Use 'diff' or 'concat'.")

            X_rows.append(pair_vec)
            y_rows.append(label)
            meta.append({
                "problem_id": problem["problem_id"],
                "difficulty": problem["difficulty"],
                "pair_index": i,
            })
            groups.append(problem_idx)

        problem_idx += 1

    if not X_rows:
        raise ValueError("No pairs found. Check that the dataset loaded correctly.")

    X      = np.stack(X_rows, axis=0).astype(np.float32)
    y      = np.array(y_rows, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)

    return X, y, meta, groups
