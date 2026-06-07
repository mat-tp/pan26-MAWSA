"""
Character n-gram features.

Extracts bigrams, trigrams, and 4-grams using HashingVectorizer so no
vocabulary fitting is required.
"""

import numpy as np
from scipy.sparse import hstack as sp_hstack
from sklearn.feature_extraction.text import HashingVectorizer

_HASH_BITS  = 12
_N_FEATURES = 1 << _HASH_BITS  # 4096


def _make_vectorizer(n):
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(n, n),
        n_features=_N_FEATURES,
        norm="l2",
        alternate_sign=False,
        lowercase=True,
        dtype=np.float32,
    )


_VECTORIZERS = {2: _make_vectorizer(2), 3: _make_vectorizer(3), 4: _make_vectorizer(4)}

NAMES_BIGRAMS   = [f"cng2_{i}" for i in range(_N_FEATURES)]
NAMES_TRIGRAMS  = [f"cng3_{i}" for i in range(_N_FEATURES)]
NAMES_FOURGRAMS = [f"cng4_{i}" for i in range(_N_FEATURES)]
NAMES = NAMES_BIGRAMS + NAMES_TRIGRAMS + NAMES_FOURGRAMS


def extract(sentence):
    """Concatenate bigram + trigram + 4-gram vectors for a single sentence."""
    sparse_vec = sp_hstack([
        _VECTORIZERS[2].transform([sentence]),
        _VECTORIZERS[3].transform([sentence]),
        _VECTORIZERS[4].transform([sentence]),
    ])
    return sparse_vec.toarray().ravel().astype(np.float32)


def extract_batch(sentences):
    """
    Vectorised batch extraction.

    Returns a sparse CSR matrix of shape (n_sentences, 3 * _N_FEATURES)
    using a single hstack per n-gram order — much faster than calling
    extract() in a loop.
    """
    mats = [vec.transform(sentences) for vec in _VECTORIZERS.values()]
    return sp_hstack(mats, format="csr")