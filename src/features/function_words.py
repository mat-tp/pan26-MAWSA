"""
Function word / stopword features.

Uses NLTK's English stopword list as a proxy for closed-class function words.
These features capture stylistic preferences that are largely topic-independent.
"""

import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z]+")


def ensure_nltk_data():
    """Download NLTK stopword corpus if not already present."""
    import nltk
    from nltk.corpus import stopwords
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def _load_function_words():
    ensure_nltk_data()
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("english"))


FUNCTION_WORDS = _load_function_words()

# Personal pronouns are reliable stylistic markers across topics
PRONOUNS = frozenset({
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
})

AUXILIARIES = frozenset({
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can", "could",
})

DETERMINERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their",
})

CONJUNCTIONS = frozenset({
    "and", "or", "but", "because", "so", "although", "however", "while", "if", "when",
    "though", "since", "until", "whereas", "nor", "yet",
})

INDIVIDUAL_WORDS = sorted(FUNCTION_WORDS)

NAMES = [
    "fw_count",
    "fw_ratio",
    "fw_unique_ratio",
    "pronoun_ratio",
    "auxiliary_ratio",
    "determiner_ratio",
    "conjunction_ratio",
]


def extract(sentence):
    """Extract aggregate function-word features for a single sentence."""
    tokens   = _TOKEN_RE.findall(sentence.lower())
    n_tokens = max(1, len(tokens))

    fw_tokens = [t for t in tokens if t in FUNCTION_WORDS]
    fw_count  = len(fw_tokens)
    aux_count = sum(1 for t in tokens if t in AUXILIARIES)
    det_count = sum(1 for t in tokens if t in DETERMINERS)
    conj_count = sum(1 for t in tokens if t in CONJUNCTIONS)

    return np.array([
        fw_count,
        fw_count / n_tokens,
        len(set(fw_tokens)) / max(1, fw_count),
        sum(1 for t in tokens if t in PRONOUNS) / n_tokens,
        aux_count / n_tokens,
        det_count / n_tokens,
        conj_count / n_tokens,
    ], dtype=np.float32)


def extract_per_word(sentence, word_list=None):
    """
    Normalized frequency vector — one feature per stopword.

    Example output features: fw_the, fw_and, fw_of, ...
    """
    if word_list is None:
        word_list = INDIVIDUAL_WORDS
    tokens   = _TOKEN_RE.findall(sentence.lower())
    n_tokens = max(1, len(tokens))
    counts   = Counter(tokens)
    return np.array([counts.get(w, 0) / n_tokens for w in word_list], dtype=np.float32)


def get_names_per_word(word_list=None):
    """Feature names corresponding to extract_per_word()."""
    if word_list is None:
        word_list = INDIVIDUAL_WORDS
    return [f"fw_{w}" for w in word_list]


def extract_batch(sentences, per_word=False):
    """
    Vectorised batch extraction.

    Args:
        sentences: list of strings
        per_word:  if True, append per-word frequency vector for each stopword
    """
    word_list = INDIVIDUAL_WORDS if per_word else None
    n_base    = len(NAMES)
    n_pw      = len(INDIVIDUAL_WORDS) if per_word else 0
    n         = len(sentences)
    out       = np.zeros((n, n_base + n_pw), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        tokens   = _TOKEN_RE.findall(sentence.lower())
        n_tokens = max(1, len(tokens))
        counts   = Counter(tokens)

        fw_tokens = [t for t in tokens if t in FUNCTION_WORDS]
        fw_count  = len(fw_tokens)
        aux_count = sum(1 for t in tokens if t in AUXILIARIES)
        det_count = sum(1 for t in tokens if t in DETERMINERS)
        conj_count = sum(1 for t in tokens if t in CONJUNCTIONS)

        out[i, 0] = fw_count
        out[i, 1] = fw_count / n_tokens
        out[i, 2] = len(set(fw_tokens)) / max(1, fw_count)
        out[i, 3] = sum(1 for t in tokens if t in PRONOUNS) / n_tokens
        out[i, 4] = aux_count / n_tokens
        out[i, 5] = det_count / n_tokens
        out[i, 6] = conj_count / n_tokens

        if per_word:
            for j, w in enumerate(INDIVIDUAL_WORDS):
                out[i, n_base + j] = counts.get(w, 0) / n_tokens

    return out