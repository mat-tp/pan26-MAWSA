"""
Pairwise dataset construction for author switch detection.

Memory design
-------------
Dense features are written to a pre-allocated np.memmap (dense.dat).
Sparse char-ngram features are stored as individual per-problem CSR files
(sparse_NNNNNN.npz) — never merged — so peak RAM is bounded to one chunk.
Metadata is stored in meta.npz + problem_meta.pkl.

Key design constraints
----------------------
- X_sparse property raises RuntimeError to prevent silent OOM.
  Use iter_chunks() for training loops or X_sparse_unsafe() when RAM allows.
- to_memory() returns compact per-problem metadata; call expand_meta() only
  when 121k per-pair dicts are genuinely needed.
- pair_dense_cols is computed from mode before memmap allocation so that
  "combined", "concat", "cosine", and "euclidean" modes are handled correctly.
- Cache hit does NOT require chunk files when n_sparse == 0.
- Chunk count is validated against problem_meta on load to detect corruption.

On-disk layout  CACHE_DIR/pairwise/<mode>_<key>/
    dense.dat           np.memmap  (n_pairs, pair_dense_cols)  float32
    sparse_NNNNNN.npz   per-problem CSR chunks (never merged)
    meta.npz            y, groups, n_dense, n_sparse, n_pairs,
                        pair_dense_cols, n_chunks (scalars)
    problem_meta.pkl    compact per-problem table (list of dicts)
"""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import pickle
import shutil
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from utils.config import CACHE_DIR, CACHE_ENABLED
from features.pipeline import FeaturePipeline


# ============================================================================
# Types
# ============================================================================

DatasetTuple = Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]


# ============================================================================
# Cache key helpers
# ============================================================================

def _compute_cache_key(problems_hash: str, feat_config_dict: dict) -> str:
    """Deterministic hash of (problem fingerprint, feature config)."""
    data_str = json.dumps(
        {"problems_hash": problems_hash, "feat_config": feat_config_dict},
        sort_keys=True,
    )
    return hashlib.sha256(data_str.encode()).hexdigest()


def _hash_problems(problems: list) -> str:
    """Fast hash of problem list — IDs + shape only, no raw text."""
    info = [
        {
            "problem_id":  p["problem_id"],
            "difficulty":  p["difficulty"],
            "n_sentences": len(p["sentences"]),
            "changes_sum": sum(p["changes"]),
        }
        for p in problems
    ]
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()


def _hash_feature_names(names: list) -> str:
    """Hash of the full feature-name list so stale caches are detected."""
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def _get_pairwise_cache_dir(cache_key: str, mode: str) -> str:
    d = os.path.join(CACHE_DIR, "pairwise", f"{mode}_{cache_key}")
    os.makedirs(d, exist_ok=True)
    return d


# ============================================================================
# Sparse-column mask
# ============================================================================

def _sparse_col_mask(feature_names: list) -> np.ndarray:
    """
    Boolean mask: True for char-ngram columns (cng2_*, cng3_*, cng4_*).
    These are ~12 288 near-zero columns — the dominant memory source.
    """
    return np.array(
        [n.startswith(("cng2_", "cng3_", "cng4_")) for n in feature_names],
        dtype=bool,
    )


# ============================================================================
# Dense column count for a given mode
# ============================================================================

def _pair_dense_cols(n_dense_feats: int, mode: str) -> int:
    """
    Return the number of output columns produced by compute_pairwise_features()
    for the dense block, given the raw feature count and the requested mode.

    This must be called BEFORE allocating the memmap so the shape is correct.
    """
    if mode == "diff":
        return n_dense_feats
    if mode == "concat":
        return n_dense_feats * 2
    if mode in ("cosine", "euclidean"):
        return 1
    if mode == "combined":
        return n_dense_feats + 2   # diff columns + cosine_distance + euclidean_distance
    raise ValueError(
        f"Unknown mode {mode!r}. "
        "Choose: 'diff', 'concat', 'cosine', 'euclidean', 'combined'."
    )


# ============================================================================
# Pairwise distance metrics  (float32, vectorized)
# ============================================================================

def compute_pairwise_features(
    left: np.ndarray,
    right: np.ndarray,
    mode: str = "diff",
) -> np.ndarray:
    """
    Compute pairwise features from two (n, F) float32 matrices.

    NOTE: when called for the *sparse* block, the caller always uses mode
    "diff" and converts the result to CSR directly — see build loop.

    Returns float32 ndarray of shape (n, pair_dense_cols(F, mode)).
    """
    l = np.asarray(left,  dtype=np.float32)
    r = np.asarray(right, dtype=np.float32)

    if mode == "diff":
        return np.abs(l - r)

    if mode == "concat":
        return np.hstack([l, r])

    if mode == "cosine":
        dot   = np.einsum("ij,ij->i", l, r)
        denom = np.linalg.norm(l, axis=1) * np.linalg.norm(r, axis=1)
        cos   = np.where(denom > 0, dot / denom, 0.0).astype(np.float32)
        return (1.0 - cos).reshape(-1, 1)

    if mode == "euclidean":
        d = l - r
        return np.sqrt(np.einsum("ij,ij->i", d, d)).astype(np.float32).reshape(-1, 1)

    if mode == "combined":
        diff_f = np.abs(l - r)
        dot    = np.einsum("ij,ij->i", l, r)
        denom  = np.linalg.norm(l, axis=1) * np.linalg.norm(r, axis=1)
        cos_d  = (1.0 - np.where(denom > 0, dot / denom, 0.0)).astype(np.float32).reshape(-1, 1)
        d2     = l - r
        euc    = np.sqrt(np.einsum("ij,ij->i", d2, d2)).astype(np.float32).reshape(-1, 1)
        return np.hstack([diff_f, cos_d, euc])

    raise ValueError(
        f"Unknown mode {mode!r}. "
        "Choose: 'diff', 'concat', 'cosine', 'euclidean', 'combined'."
    )


# ============================================================================
# Per-pair meta expansion helper
# ============================================================================

def expand_meta(problem_meta: list, y: np.ndarray) -> list:
    """
    Expand compact per-problem metadata into one dict per pair.

    Call this only when you genuinely need 121k dicts; most callers should
    work with problem_meta directly.
    """
    meta = []
    for rec in problem_meta:
        pid   = rec["problem_id"]
        diff  = rec["difficulty"]
        start = rec["start_row"]
        end   = rec["end_row"]
        for pair_idx in range(end - start):
            global_row = start + pair_idx
            meta.append({
                "problem_id": pid,
                "difficulty": diff,
                "pair_idx":   pair_idx,
                "is_switch":  bool(y[global_row]),
            })
    return meta


# ============================================================================
# PairwiseDataset — iterator + lazy materialisation
# ============================================================================

class PairwiseDataset:
    """
    Lazy wrapper around the on-disk pairwise dataset.

    Sparse data lives in individual per-problem chunk files.
    No merged sparse matrix is ever written, so RAM is always bounded.

    Iterate chunk-by-chunk for partial_fit / external-memory training:

        for X_d, X_s, y_c in dataset.iter_chunks(chunk_size=5000):
            model.partial_fit(
                scipy.sparse.hstack([X_d, X_s]),
                y_c, classes=[0, 1]
            )

    Or load everything at once (only if you have the RAM):

        X_dense, y, problem_meta, groups = dataset.to_memory()
        # If you also need sparse:
        X_sparse = dataset.X_sparse_unsafe()
    """

    def __init__(
        self,
        cache_dir: str,
        n_pairs: int,
        n_dense: int,          # raw feature count (input side)
        n_sparse: int,         # raw sparse feature count
        pair_dense_cols: int,  # actual output columns in dense.dat
        y: np.ndarray,
        groups: np.ndarray,
        problem_meta: list,    # one dict per *problem* (not per pair)
    ):
        self.cache_dir       = cache_dir
        self.n_pairs         = n_pairs
        self.n_dense         = n_dense
        self.n_sparse        = n_sparse
        self.pair_dense_cols = pair_dense_cols
        self.y               = y
        self.groups          = groups
        self.problem_meta    = problem_meta

        self._fitted = False

        self._dense_path = os.path.join(cache_dir, "dense.dat")

        # Sorted list of per-problem sparse chunk files.
        # When n_sparse == 0 this list is legitimately empty.
        self._chunk_files: list = sorted(
            glob.glob(os.path.join(cache_dir, "sparse_[0-9]*.npz"))
        )

            # Only enforced when sparse features exist; an empty list is valid
        # when n_sparse == 0.
        if n_sparse > 0 and len(self._chunk_files) != len(problem_meta):
            raise RuntimeError(
                f"Sparse chunk count ({len(self._chunk_files)}) does not match "
                f"problem_meta length ({len(problem_meta)}). "
                "The cache may be partially written or corrupted. "
                "Call clear_pairwise_cache() and rebuild."
            )

    # ------------------------------------------------------------------
    # Dense accessor (always memmap — OS pages on demand)
    # ------------------------------------------------------------------

    @property
    def X_dense(self) -> np.memmap:
        return np.memmap(
            self._dense_path, dtype=np.float32, mode="r",
            shape=(self.n_pairs, self.pair_dense_cols),
        )

    # ------------------------------------------------------------------
    # Sparse accessor — Fix 2: refuse silent OOM; provide escape hatch
    # ------------------------------------------------------------------

    @property
    def X_sparse(self) -> None:
        raise RuntimeError(
            "Calling dataset.X_sparse would load the entire sparse matrix "
            f"({self.n_pairs} × {self.n_sparse} float32) into RAM, which "
            "may be several GB.\n"
            "Use dataset.iter_chunks() for bounded-memory access, or call "
            "dataset.X_sparse_unsafe() if you have confirmed you have "
            "sufficient RAM."
        )

    def X_sparse_unsafe(self) -> csr_matrix:
        """
        Load and vertically stack all sparse chunks into one CSR matrix.

        Collects all chunks into a list first, then calls sp_vstack once.
        This is O(n) in chunk count rather than the O(n²) pattern of
        repeatedly doing ``result = sp_vstack([result, chunk])``, which
        re-allocates the growing result on every iteration.

        Peak RAM ≈ final_matrix + the last chunk loaded (SciPy vstack
        builds a new matrix from the COO data of all inputs).

        This can still consume several GB for large datasets.
        Prefer iter_chunks() for training loops.
        """
        from scipy.sparse import vstack as sp_vstack
        if not self._chunk_files:
            return csr_matrix((self.n_pairs, self.n_sparse), dtype=np.float32)
        chunks = [load_npz(path) for path in self._chunk_files]
        return sp_vstack(chunks, format="csr")

    # ------------------------------------------------------------------
    # Chunk iterator — truly bounded memory
    # ------------------------------------------------------------------

    def iter_chunks(
        self, chunk_size: int = 5_000
    ) -> Iterator[Tuple[np.ndarray, csr_matrix, np.ndarray]]:
        """
        Yield (X_dense_chunk, X_sparse_chunk, y_chunk) slices.

        Dense slices come from memmap (zero extra RAM beyond one slice).
        Sparse slices come from loading one chunk file at a time, then
        further slicing if the problem has more rows than chunk_size.
        Peak RAM = one dense slice + one sparse chunk file.

        When n_sparse == 0 (no char-ngrams) there are no chunk files.
        A dedicated dense-only path still yields every batch with a
        zero-column sparse placeholder so the caller API is uniform.
        """
        X_d = self.X_dense

        if self.n_sparse == 0:
            for start in range(0, self.n_pairs, chunk_size):
                end = min(start + chunk_size, self.n_pairs)
                yield (
                    X_d[start:end],
                    csr_matrix((end - start, 0), dtype=np.float32),
                    self.y[start:end],
                )
            return

        # Normal path — one chunk file per problem
        dense_row = 0
        for prob_idx, chunk_path in enumerate(self._chunk_files):
            rec       = self.problem_meta[prob_idx]
            prob_rows = rec["end_row"] - rec["start_row"]
            X_s_full  = load_npz(chunk_path)   # one problem at a time

            # A single problem's chunk may itself exceed chunk_size
            for inner_start in range(0, prob_rows, chunk_size):
                inner_end    = min(inner_start + chunk_size, prob_rows)
                global_start = dense_row + inner_start
                global_end   = dense_row + inner_end
                yield (
                    X_d[global_start:global_end],
                    X_s_full[inner_start:inner_end],
                    self.y[global_start:global_end],
                )

            del X_s_full
            gc.collect()
            dense_row += prob_rows

    # ------------------------------------------------------------------
    # Full materialisation — Fix 3: return problem_meta, not 121k dicts
    # ------------------------------------------------------------------

    def to_memory(self) -> DatasetTuple:
        """
        Load dense data into RAM. Sparse is NOT loaded here.

        Returns (X_dense, y, problem_meta, groups).

        problem_meta is a list of per-problem dicts:
            [{problem_id, difficulty, start_row, end_row}, …]

        If you need per-pair expansion, call:
            expand_meta(problem_meta, y)
        If you need sparse, call:
            dataset.X_sparse_unsafe()
        """
        X_dense = np.array(self.X_dense)  # copy memmap → contiguous RAM
        return X_dense, self.y, self.problem_meta, self.groups

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_pairs

    def __repr__(self) -> str:
        return (
            f"PairwiseDataset("
            f"n_pairs={self.n_pairs}, "
            f"pair_dense_cols={self.pair_dense_cols}, "
            f"n_sparse={self.n_sparse}, "
            f"problems={len(self.problem_meta)}, "
            f"chunks={len(self._chunk_files)})"
        )


# ============================================================================
# Sentence-feature cache helpers
# ============================================================================

def _build_sentence_cache_index(problems: list, problems_hash: str) -> tuple:
    sent_cache_dir = os.path.join(CACHE_DIR, "sentence_features", problems_hash[:16])
    index = {}
    if os.path.isdir(sent_cache_dir):
        for p in problems:
            ck   = f"{p['difficulty']}_{p['problem_id']}"
            path = os.path.join(sent_cache_dir, f"{ck}.npy")
            if os.path.exists(path):
                index[ck] = path
    return sent_cache_dir, index


def _flush_sentence_cache(sent_cache_dir: str, new_entries: dict) -> None:
    if not new_entries:
        return
    os.makedirs(sent_cache_dir, exist_ok=True)
    for ck, feats in new_entries.items():
        path = os.path.join(sent_cache_dir, f"{ck}.npy")
        if not os.path.exists(path):
            np.save(path, feats)
    print(f"[pairwise] Cached {len(new_entries)} sentence-feature matrices → {sent_cache_dir}")


# ============================================================================
# Main builder

def build_pairwise_dataset(
    problems: list,
    feature_pipeline: Optional[FeaturePipeline] = None,
    min_sentences: int = 3,
    mode: str = "diff",
    use_cache: bool = True,
    n_jobs: int = 1,
    cache_sentence_features: bool = True,
) -> PairwiseDataset:
    """
    Build the pairwise training dataset without accumulating features in RAM.

    Returns a PairwiseDataset which supports:
      - Chunk iteration:   for X_d, X_s, y_c in ds.iter_chunks(chunk_size): ...
      - Full dense load:   X_d, y, meta, groups = ds.to_memory()
      - Direct attribute:  ds.X_dense, ds.y, ds.groups, ds.problem_meta
      - Unsafe full load:  X_s = ds.X_sparse_unsafe()

    chunk_size is intentionally NOT a parameter here — it belongs to
    iter_chunks() so the caller can choose the right size at training time.

    Memory strategy
    ---------------
    Dense features:
        Pre-allocated np.memmap with the correct shape for the chosen mode.
    Sparse features (char-ngrams):
        Each problem's sparse block written directly as CSR to
        sparse_NNNNNN.npz. No merge step — peak RAM = 1 chunk.
        Sparse diff computed with CSR arithmetic — no dense intermediate.
    Metadata:
        One compact record per *problem* (not per pair).
    X_sparse property:
        Raises RuntimeError to prevent silent OOM.
    to_memory():
        Returns problem_meta directly, no per-pair dict expansion.
    Cache key:
        Includes SHA-256 of feature_names to catch silent feature changes.
    Cache hit with n_sparse == 0:
        Chunk-file presence is not required for a cache hit.
    Integrity check:
        __init__ validates chunk count == problem count when n_sparse > 0.
    """
    t0 = time.time()

    if feature_pipeline is None:
        feature_pipeline = FeaturePipeline()

    # ── Column layout ──────────────────────────────────────────────────────
    all_names    = feature_pipeline.feature_names
    sparse_mask  = _sparse_col_mask(all_names)
    n_dense      = int((~sparse_mask).sum())
    n_sparse     = int(sparse_mask.sum())

    pdc = _pair_dense_cols(n_dense, mode)

    # ── Cache key ──────────────────────────────────────────────────────────
    problems_hash = _hash_problems(problems)
    feat_config   = {
        "groups":             feature_pipeline.groups,
        "use_per_word_fw":    feature_pipeline.use_per_word_fw,
        "mode":               mode,
        "min_sentences":      min_sentences,
        "ngram_enabled":      feature_pipeline.ngram_enabled,
        "ngram_max_n":        (
            getattr(feature_pipeline._ngram_extractor, "max_n", None)
            if feature_pipeline.ngram_enabled else None
        ),
        "feature_names_hash": _hash_feature_names(all_names),
    }
    cache_key = _compute_cache_key(problems_hash, feat_config)
    cache_dir = _get_pairwise_cache_dir(cache_key, mode)

    dense_path = os.path.join(cache_dir, "dense.dat")
    meta_path  = os.path.join(cache_dir, "meta.npz")

    # ── Try pairwise cache ─────────────────────────────────────────────────
    # chunk files are written and the cache is still valid.
    # PairwiseDataset.__init__ handles an empty _chunk_files list correctly.
    if use_cache and CACHE_ENABLED:
        if os.path.exists(dense_path) and os.path.exists(meta_path):
            print(f"[pairwise] Cache hit → {cache_dir}")
            return _load_dataset_from_cache(cache_dir, dense_path, meta_path)

    print("[pairwise] Building pairwise dataset from scratch …")

    # ── Filter valid problems ──────────────────────────────────────────────
    valid       = [p for p in problems if len(p["sentences"]) >= min_sentences]
    n_pairs_est = sum(len(p["sentences"]) - 1 for p in valid)

    print(
        f"[pairwise] {len(valid)} valid problems → ~{n_pairs_est} pairs | "
        f"{n_dense} dense feats ({pdc} output cols) + {n_sparse} sparse feats"
    )
    dense_mb = n_pairs_est * pdc * 4 / 1e6
    print(f"[pairwise] Dense memmap ≈ {dense_mb:.1f} MB on disk")

    # ── Fit n-gram models ──────────────────────────────────────────────────
    # Skip refitting - pipeline should already be fitted by caller
    # Skip refitting - pipeline should already be fitted by caller
    if feature_pipeline.ngram_enabled and not feature_pipeline.is_fitted:
        # If not fitted, fit it now (but this shouldn't happen)
        all_sents = [s for p in valid for s in p["sentences"]]
        print(f"[pairwise] WARNING: Pipeline not fitted, fitting now...")
        feature_pipeline.fit(all_sents)

    # ── Sentence-feature cache ─────────────────────────────────────────────
    use_sc = use_cache and CACHE_ENABLED and cache_sentence_features
    sent_cache_dir, sent_index = (
        _build_sentence_cache_index(valid, problems_hash) if use_sc else ("", {})
    )
    new_sent: dict = {}

    # ── Pre-allocate dense memmap with correct shape ──────────────
    X_mm = np.memmap(
        dense_path, dtype=np.float32, mode="w+",
        shape=(n_pairs_est, pdc),
    )

    # ── Compact metadata ───────────────────────────────────────────────────
    y_buf      = np.empty(n_pairs_est, dtype=np.int8)
    groups_buf = np.empty(n_pairs_est, dtype=np.int32)
    problem_meta: list = []

    row = 0

    # ── Main loop — one problem at a time ─────────────────────────────────
    for prob_idx, problem in enumerate(valid):
        pid       = problem["problem_id"]
        diff      = problem["difficulty"]
        sentences = problem["sentences"]
        changes   = problem["changes"]
        ck        = f"{diff}_{pid}"
        n_sents   = len(sentences)
        n_pairs_p = n_sents - 1

        # Sentence features — keep dense and sparse separate throughout.
        # This avoids the full-width hstack that was the original memory problem.
        if use_sc and ck in sent_index:
            # Cache stores only the dense block; sparse must be re-extracted.
            # (Storing sparse as .npy would require toarray(), defeating the point.)
            dense_mat = np.load(sent_index[ck]).astype(np.float32, copy=False)
            _, sparse_mat = feature_pipeline.extract_split(sentences)
        else:
            dense_mat, sparse_mat = feature_pipeline.extract_split(sentences)
            if use_sc:
                # Cache only the dense block — it's small and contiguous.
                new_sent[ck] = dense_mat

        left_dense  = dense_mat[:-1]   # (n_pairs_p, n_dense)
        right_dense = dense_mat[1:]
        left_sp     = sparse_mat[:-1]  # already CSR slices
        right_sp    = sparse_mat[1:]

        # Dense columns → write to memmap
        pair_dense = compute_pairwise_features(left_dense, right_dense, mode)
        X_mm[row: row + n_pairs_p] = pair_dense

        # Sparse columns → compute diff entirely in sparse arithmetic.
        # Slices from a CSR matrix are already CSR; tocsr() guarantees format.
        # abs() the non-zero values in-place via .data; no dense intermediate.
        left_sp  = left_sp.tocsr().astype(np.float32, copy=False)
        right_sp = right_sp.tocsr().astype(np.float32, copy=False)
        pair_sparse = (left_sp - right_sp).tocsr()
        pair_sparse.data = np.abs(pair_sparse.data)    # in-place, stays sparse
        chunk_path = os.path.join(cache_dir, f"sparse_{prob_idx:06d}.npz")
        save_npz(chunk_path, pair_sparse)

        # Labels / groups
        y_buf[row: row + n_pairs_p]      = np.asarray(changes, dtype=np.int8)
        groups_buf[row: row + n_pairs_p] = prob_idx

        problem_meta.append({
            "problem_id": pid,
            "difficulty": diff,
            "start_row":  row,
            "end_row":    row + n_pairs_p,
        })

        row += n_pairs_p

        del dense_mat, sparse_mat, left_dense, right_dense, pair_dense, left_sp, right_sp, pair_sparse
        if prob_idx % 100 == 0:
            gc.collect()

        if (prob_idx + 1) % 50 == 0 or prob_idx + 1 == len(valid):
            print(f"[pairwise] Processed {prob_idx + 1}/{len(valid)} problems")

    n_pairs = row

    # ── Flush dense memmap ─────────────────────────────────────────────────
    X_mm.flush()
    del X_mm
    gc.collect()

    # ── Save metadata (no merged sparse file — Fix 1) ─────────────────────
    y      = y_buf[:n_pairs]
    groups = groups_buf[:n_pairs]
    np.savez(
        meta_path,
        n_pairs         = np.array(n_pairs,              dtype=np.int64),
        n_dense         = np.array(n_dense,              dtype=np.int32),
        n_sparse        = np.array(n_sparse,             dtype=np.int32),
        pair_dense_cols = np.array(pdc,                  dtype=np.int32),
        n_chunks        = np.array(len(problem_meta),    dtype=np.int32),
        y               = y,
        groups          = groups,
    )
    with open(os.path.join(cache_dir, "problem_meta.pkl"), "wb") as fh:
        pickle.dump(problem_meta, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Save sentence feature cache ────────────────────────────────────────
    if use_sc:
        _flush_sentence_cache(sent_cache_dir, new_sent)
    del new_sent

    elapsed = time.time() - t0
    print(
        f"[pairwise] Built in {elapsed:.1f}s — "
        f"{n_pairs} pairs | dense {n_dense} feats → {pdc} cols | "
        f"sparse {n_sparse} cols | {len(problem_meta)} chunk files"
    )
    class_0 = int((y == 0).sum())
    class_1 = int((y == 1).sum())
    print(f"[pairwise] Class distribution: {class_0} same / {class_1} switch")

    return PairwiseDataset(
        cache_dir       = cache_dir,
        n_pairs         = n_pairs,
        n_dense         = n_dense,
        n_sparse        = n_sparse,
        pair_dense_cols = pdc,
        y               = y,
        groups          = groups,
        problem_meta    = problem_meta,
    )


def _load_dataset_from_cache(
    cache_dir: str,
    dense_path: str,
    meta_path: str,
) -> PairwiseDataset:
    """Reconstruct a PairwiseDataset from existing cache files."""
    meta_np         = np.load(meta_path, allow_pickle=False)
    n_pairs         = int(meta_np["n_pairs"])
    n_dense         = int(meta_np["n_dense"])
    n_sparse        = int(meta_np["n_sparse"])
    # Graceful degradation for caches written before Fix 4
    pair_dense_cols = (
        int(meta_np["pair_dense_cols"])
        if "pair_dense_cols" in meta_np
        else n_dense
    )
    y      = meta_np["y"]
    groups = meta_np["groups"]

    prob_meta_path = os.path.join(cache_dir, "problem_meta.pkl")
    if os.path.exists(prob_meta_path):
        with open(prob_meta_path, "rb") as fh:
            problem_meta = pickle.load(fh)
    else:
        problem_meta = []

    # n_chunks was added in v6; fall back gracefully for older caches.
    if "n_chunks" in meta_np and n_sparse > 0:
        expected_chunks = int(meta_np["n_chunks"])
        found_chunks    = len(sorted(
            glob.glob(os.path.join(cache_dir, "sparse_[0-9]*.npz"))
        ))
        if found_chunks != expected_chunks:
            raise RuntimeError(
                f"[pairwise] Cache corruption detected in {cache_dir}: "
                f"expected {expected_chunks} sparse chunk files, "
                f"found {found_chunks}. "
                "Call clear_pairwise_cache() and rebuild."
            )
        # Also verify problem_meta length agrees with n_chunks so that a
        # truncated pickle can't silently misalign rows with chunk files.
        if problem_meta and len(problem_meta) != expected_chunks:
            raise RuntimeError(
                f"[pairwise] Cache corruption detected in {cache_dir}: "
                f"problem_meta has {len(problem_meta)} entries but "
                f"n_chunks == {expected_chunks}. "
                "Call clear_pairwise_cache() and rebuild."
            )

    print(
        f"[pairwise] Cache loaded — {n_pairs} pairs | "
        f"dense {n_dense} feats → {pair_dense_cols} cols | sparse {n_sparse}"
    )
    return PairwiseDataset(
        cache_dir       = cache_dir,
        n_pairs         = n_pairs,
        n_dense         = n_dense,
        n_sparse        = n_sparse,
        pair_dense_cols = pair_dense_cols,
        y               = y,
        groups          = groups,
        problem_meta    = problem_meta,
    )


# ============================================================================
# Convenience builder — raw text pairs  (no caching, inference only)
# ============================================================================

def build_pairwise_from_texts(
    texts1: list,
    texts2: list,
    labels: Optional[list] = None,
    feature_pipeline: Optional[FeaturePipeline] = None,
    mode: str = "diff",
) -> tuple:
    """
    Build a pairwise matrix from two parallel text lists.
    Intended for small inference batches — no caching.

    Returns (X_dense, X_sparse, y, meta).
    """
    if feature_pipeline is None:
        feature_pipeline = FeaturePipeline()
    if feature_pipeline.ngram_enabled and not feature_pipeline._fitted:
        feature_pipeline.fit(texts1 + texts2)

    all_names   = feature_pipeline.feature_names
    sparse_mask = _sparse_col_mask(all_names)

    f1_dense, f1_sparse = feature_pipeline.extract_split(texts1)
    f2_dense, f2_sparse = feature_pipeline.extract_split(texts2)

    X_dense = compute_pairwise_features(
        f1_dense.astype(np.float32, copy=False),
        f2_dense.astype(np.float32, copy=False),
        mode,
    )
    f1_sp    = f1_sparse.tocsr().astype(np.float32, copy=False)
    f2_sp    = f2_sparse.tocsr().astype(np.float32, copy=False)
    X_sparse = (f1_sp - f2_sp).tocsr()
    X_sparse.data = np.abs(X_sparse.data)   # in-place abs on non-zeros only
    meta = [
        {"pair_idx": i, "sentence1": texts1[i][:100], "sentence2": texts2[i][:100]}
        for i in range(len(texts1))
    ]
    y = np.array(labels, dtype=np.int8) if labels is not None else None
    return X_dense, X_sparse, y, meta


# ============================================================================
# Feature name helpers
# ============================================================================

def get_pairwise_feature_names(
    feature_pipeline: FeaturePipeline,
    mode: str = "diff",
) -> tuple:
    """
    Return (dense_names, sparse_names) for the pairwise feature split.
    Sparse names always reflect the "diff" operation.
    """
    all_names    = feature_pipeline.get_feature_names()
    sparse_mask  = _sparse_col_mask(all_names)
    dense_names  = [n for n, s in zip(all_names, sparse_mask) if not s]
    sparse_names = [n for n, s in zip(all_names, sparse_mask) if s]

    def _pref(names, tag):
        return [f"{tag}_{n}" for n in names]

    sparse_out = _pref(sparse_names, "diff")

    if mode == "diff":
        return _pref(dense_names, "diff"), sparse_out
    if mode == "concat":
        d = _pref(dense_names, "left") + _pref(dense_names, "right")
        return d, sparse_out
    if mode == "cosine":
        return ["cosine_distance"], sparse_out
    if mode == "euclidean":
        return ["euclidean_distance"], sparse_out
    if mode == "combined":
        d = _pref(dense_names, "diff") + ["cosine_distance", "euclidean_distance"]
        return d, sparse_out

    raise ValueError(f"Unknown mode: {mode!r}")


# ============================================================================
# Cache management
# ============================================================================

def clear_pairwise_cache(mode: Optional[str] = None) -> None:
    """Clear pairwise feature cache (all modes or a specific one)."""
    cache_root = os.path.join(CACHE_DIR, "pairwise")
    if not os.path.exists(cache_root):
        print("[cache] No pairwise cache found")
        return
    if mode:
        pattern = os.path.join(cache_root, f"{mode}_*")
        dirs    = glob.glob(pattern)
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
        print(f"[cache] Cleared {len(dirs)} cache dir(s) for mode '{mode}'")
    else:
        shutil.rmtree(cache_root, ignore_errors=True)
        print("[cache] Cleared all pairwise cache")