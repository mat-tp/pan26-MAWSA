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


def extract_batch(sentences):
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
    
    # PCA dimensionality reduction: 384 → 128 dimensions (66% memory savings)
    # Threshold: only apply if embeddings are high-dimensional (safety check)
    if embeddings.shape[1] > 256:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=EMBEDDING_DIM_REDUCED, random_state=42)
            embeddings = pca.fit_transform(embeddings).astype(np.float32)
            print(f"[embeddings] Reduced from {EMBEDDING_DIM} to {EMBEDDING_DIM_REDUCED} dims via PCA")
        except Exception as e:
            print(f"[embeddings] PCA reduction failed ({e}), using full embeddings")
    
    return embeddings
