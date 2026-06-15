"""
Optional sentence-embedding features for author-style comparison.
"""

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_DIM_REDUCED = 128  # After PCA dimensionality reduction

_model = None
_pca = None  # Global fitted PCA model


def _get_model():
    global _model
    if _model is None:
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install it with `pip install sentence-transformers`."
            )
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_feature_names():
    # After PCA reduction: 384 → 128 dimensions
    return [f"emb_{i}" for i in range(EMBEDDING_DIM_REDUCED)]


def fit(sentences):
    """
    Fit the PCA transformation on training sentences.
    This MUST be called once before extract_batch() to ensure consistent dimensions.
    """
    global _pca
    if not sentences:
        print("[embeddings] No sentences to fit PCA")
        return
    
    print(f"[embeddings] Fitting PCA on {len(sentences)} sentences...")
    model = _get_model()
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=False,
    )
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    try:
        from sklearn.decomposition import PCA
        _pca = PCA(n_components=EMBEDDING_DIM_REDUCED, random_state=42)
        _pca.fit(embeddings)
        explained_var = _pca.explained_variance_ratio_.sum()
        print(f"[embeddings] PCA fitted: {EMBEDDING_DIM} → {EMBEDDING_DIM_REDUCED} dims "
              f"(explains {explained_var:.1%} variance)")
    except Exception as e:
        print(f"[embeddings] PCA fit failed ({e}), will use full embeddings")
        _pca = None


def extract_batch(sentences):
    """
    Extract embeddings for a batch of sentences.
    Uses the global fitted PCA model if available.
    """
    global _pca
    if not sentences:
        return np.zeros((0, EMBEDDING_DIM_REDUCED), dtype=np.float32)
    
    model = _get_model()
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=64,
        normalize_embeddings=False,
    )
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Use the globally fitted PCA model
    if _pca is not None:
        embeddings = _pca.transform(embeddings).astype(np.float32)
    else:
        # Fallback: if PCA wasn't fitted, use full embeddings
        print(f"[embeddings] Warning: using full {EMBEDDING_DIM} dims (PCA not fitted)")
        if embeddings.shape[1] != EMBEDDING_DIM_REDUCED:
            # Still try to reduce if not fitted (shouldn't happen normally)
            if embeddings.shape[1] == EMBEDDING_DIM:
                # Just take first 128 dims as emergency fallback
                embeddings = embeddings[:, :EMBEDDING_DIM_REDUCED]
    
    return embeddings
