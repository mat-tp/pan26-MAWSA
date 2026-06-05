"""
Character n-gram features.

Extracts:
    - Character bigrams
    - Character trigrams
    - Character 4-grams

Uses HashingVectorizer so no vocabulary fitting is required.
"""

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer

_HASH_BITS = 12
_N_FEATURES = 1 << _HASH_BITS


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


_VECTORIZERS = {
    2: _make_vectorizer(2),
    3: _make_vectorizer(3),
    4: _make_vectorizer(4),
}


NAMES_BIGRAMS = [f"cng2_{i}" for i in range(_N_FEATURES)]
NAMES_TRIGRAMS = [f"cng3_{i}" for i in range(_N_FEATURES)]
NAMES_FOURGRAMS = [f"cng4_{i}" for i in range(_N_FEATURES)]

NAMES = (
    NAMES_BIGRAMS
    + NAMES_TRIGRAMS
    + NAMES_FOURGRAMS
)


def _transform(sentence, n):
    return (
        _VECTORIZERS[n]
        .transform([sentence])
        .toarray()
        .ravel()
        .astype(np.float32)
    )


def extract_bigrams(sentence):
    return _transform(sentence, 2)


def extract_trigrams(sentence):
    return _transform(sentence, 3)


def extract_fourgrams(sentence):
    return _transform(sentence, 4)


def extract(sentence):
    """
    Concatenate:
        bigrams + trigrams + fourgrams

    Returns:
        np.ndarray(float32)
    """

    sparse_vec = hstack(
        [
            _VECTORIZERS[2].transform([sentence]),
            _VECTORIZERS[3].transform([sentence]),
            _VECTORIZERS[4].transform([sentence]),
        ]
    )

    return sparse_vec.toarray().ravel().astype(np.float32)

def extract_batch(sentences):
    """
    Vectorised batch extraction using the sparse HashingVectorizer transform.
    This is significantly faster than calling extract() in a loop because it
    uses a single sparse matrix hstack per n-gram order.
    """
    from scipy.sparse import vstack as sp_vstack

    mats = []
    for n, vec in _VECTORIZERS.items():
        mats.append(vec.transform(sentences))
    # hstack each n-gram matrix, then vstack would be wrong;
    # each vec.transform returns (n_sents, n_features) so we hstack the three
    # per-sentence result — but they're already in (n_sents, n_feats) shape.
    from scipy.sparse import hstack as sp_hstack
    combined = sp_hstack(mats)
    return combined.toarray().astype(np.float32)