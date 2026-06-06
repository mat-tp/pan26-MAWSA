"""
Function word / stopword features.

Uses NLTK's English stopword list as a proxy for
closed-class function words.

These features capture stylistic preferences that
are largely topic-independent.
"""

import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")


def ensure_nltk_data():
    """
    Download NLTK stopword corpus if needed.
    """

    import nltk
    from nltk.corpus import stopwords

    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def _load_function_words():
    ensure_nltk_data()

    from nltk.corpus import stopwords

    return frozenset(
        stopwords.words("english")
    )


FUNCTION_WORDS = _load_function_words()

# Personal pronouns remain useful stylistic indicators
PRONOUNS = frozenset({
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
})

INDIVIDUAL_WORDS = sorted(FUNCTION_WORDS)

NAMES = [
    "fw_count",
    "fw_ratio",
    "fw_unique_ratio",
    "pronoun_ratio",
]


def extract(sentence):
    """
    Aggregate function-word features.

    Returns:
        np.ndarray(float32)
    """

    tokens = _TOKEN_RE.findall(
        sentence.lower()
    )

    n_tokens = max(1, len(tokens))

    fw_tokens = [
        t
        for t in tokens
        if t in FUNCTION_WORDS
    ]

    fw_count = len(fw_tokens)

    fw_ratio = fw_count / n_tokens

    fw_unique_ratio = (
        len(set(fw_tokens))
        / max(1, fw_count)
    )

    pronoun_ratio = (
        sum(
            1
            for t in tokens
            if t in PRONOUNS
        )
        / n_tokens
    )

    return np.array(
        [
            fw_count,
            fw_ratio,
            fw_unique_ratio,
            pronoun_ratio,
        ],
        dtype=np.float32,
    )


def extract_per_word(
    sentence,
    word_list=None,
):
    """
    Normalized frequency vector.

    Returns one feature per stopword.

    Example:
        fw_the
        fw_and
        fw_of
        ...
    """

    if word_list is None:
        word_list = INDIVIDUAL_WORDS

    tokens = _TOKEN_RE.findall(
        sentence.lower()
    )

    n_tokens = max(1, len(tokens))

    counts = Counter(tokens)

    return np.array(
        [
            counts.get(w, 0)
            / n_tokens
            for w in word_list
        ],
        dtype=np.float32,
    )


def get_names_per_word(
    word_list=None,
):
    """
    Feature names corresponding to
    extract_per_word().
    """

    if word_list is None:
        word_list = INDIVIDUAL_WORDS

    return [
        f"fw_{w}"
        for w in word_list
    ]

def extract_batch(sentences, per_word=False):
    """
    Vectorised batch extraction.

    Pre-allocates the output array and fills it row-by-row, eliminating
    the per-sentence list-append and final vstack overhead.

    Args:
        sentences: list of strings
        per_word:  if True, append per-word frequency vector
    """
    word_list   = INDIVIDUAL_WORDS if per_word else None
    n_base      = len(NAMES)
    n_pw        = len(INDIVIDUAL_WORDS) if per_word else 0
    n_cols      = n_base + n_pw
    n           = len(sentences)
    out         = np.zeros((n, n_cols), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        tokens   = _TOKEN_RE.findall(sentence.lower())
        n_tokens = max(1, len(tokens))
        counts   = Counter(tokens)

        fw_tokens       = [t for t in tokens if t in FUNCTION_WORDS]
        fw_count        = len(fw_tokens)
        fw_ratio        = fw_count / n_tokens
        fw_unique_ratio = len(set(fw_tokens)) / max(1, fw_count)
        pronoun_ratio   = sum(1 for t in tokens if t in PRONOUNS) / n_tokens

        out[i, 0] = fw_count
        out[i, 1] = fw_ratio
        out[i, 2] = fw_unique_ratio
        out[i, 3] = pronoun_ratio

        if per_word:
            for j, w in enumerate(INDIVIDUAL_WORDS):
                out[i, n_base + j] = counts.get(w, 0) / n_tokens

    return out