"""
Character n-gram features (bigrams, trigrams, 4-grams).

Character n-grams capture sub-word stylistic patterns such as preferred
affixes, spacing habits around punctuation, and character-level vocabulary.
They are robust to spelling variation and work well at the sentence level.

We use sklearn's HashingVectorizer to avoid fitting a vocabulary — the
feature space is fixed and deterministic, which is important for TIRA
reproducibility. The hash space is large enough to avoid significant
collisions for typical sentence lengths.

For ablation studies, each n-gram size can be used independently.
"""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Hash space per n-gram size.  2^14 = 16 384 buckets per size is more than
# enough for sentence-level text while keeping memory low.
_HASH_BITS = 12   # 2^12 = 4 096 buckets per n-gram size
_N_FEATURES = 2 ** _HASH_BITS


def _make_vectorizer(ngram_range):
    return HashingVectorizer(
        analyzer="char_wb",       # char_wb pads words with spaces — more informative
        ngram_range=ngram_range,
        n_features=_N_FEATURES,
        norm="l2",                 # normalise so sentence length doesn't dominate
        alternate_sign=False,      # keep all values positive
        lowercase=True,
    )


_VEC_BIGRAMS   = _make_vectorizer((2, 2))
_VEC_TRIGRAMS  = _make_vectorizer((3, 3))
_VEC_FOURGRAMS = _make_vectorizer((4, 4))

# Feature names for each group (used by the pipeline for ablation labels)
NAMES_BIGRAMS   = [f"cng2_{i}" for i in range(_N_FEATURES)]
NAMES_TRIGRAMS  = [f"cng3_{i}" for i in range(_N_FEATURES)]
NAMES_FOURGRAMS = [f"cng4_{i}" for i in range(_N_FEATURES)]
NAMES = NAMES_BIGRAMS + NAMES_TRIGRAMS + NAMES_FOURGRAMS


def extract(sentence):
    """
    Return a 1-D float32 array of character n-gram features.
    Concatenates bigram, trigram, and 4-gram vectors.
    """
    bg  = _VEC_BIGRAMS.transform([sentence]).toarray()[0]
    tg  = _VEC_TRIGRAMS.transform([sentence]).toarray()[0]
    fg  = _VEC_FOURGRAMS.transform([sentence]).toarray()[0]
    return np.concatenate([bg, tg, fg]).astype(np.float32)


def extract_bigrams(sentence):
    return _VEC_BIGRAMS.transform([sentence]).toarray()[0].astype(np.float32)


def extract_trigrams(sentence):
    return _VEC_TRIGRAMS.transform([sentence]).toarray()[0].astype(np.float32)


def extract_fourgrams(sentence):
    return _VEC_FOURGRAMS.transform([sentence]).toarray()[0].astype(np.float32)
