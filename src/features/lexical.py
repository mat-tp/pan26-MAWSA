"""
Lexical features for sentence-level stylometry.
"""

import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z]+")

NAMES = [
    "n_chars",
    "n_tokens",
    "avg_word_len",
    "std_word_len",
    "ttr",
    "hapax_ratio",
    "simpson_d",
    "wl_short",
    "wl_medium",
    "wl_long",
    "wl_very_long",
]


def extract(sentence):
    """Extract lexical features for a single sentence."""
    words = _TOKEN_RE.findall(sentence)
    if not words:
        return np.zeros(len(NAMES), dtype=np.float32)

    n_tokens = len(words)
    lengths  = np.array([len(w) for w in words], dtype=np.float32)
    lower    = [w.lower() for w in words]
    counts   = Counter(lower)
    vocab    = len(counts)

    return np.array([
        len(sentence),
        n_tokens,
        lengths.mean(),
        lengths.std(),
        vocab / n_tokens,
        sum(c == 1 for c in counts.values()) / n_tokens,
        sum(c * (c - 1) for c in counts.values()) / max(1, n_tokens * (n_tokens - 1)),
        np.mean(lengths <= 3),
        np.mean((lengths >= 4) & (lengths <= 6)),
        np.mean((lengths >= 7) & (lengths <= 9)),
        np.mean(lengths >= 10),
    ], dtype=np.float32)


def extract_batch(sentences):
    """Vectorised batch extraction over a list of sentences."""
    n   = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        words = _TOKEN_RE.findall(sentence)
        out[i, 0] = len(sentence)

        if not words:
            out[i, 1] = 1
            continue

        n_tokens = len(words)
        lengths  = np.array([len(w) for w in words], dtype=np.float32)
        lower    = [w.lower() for w in words]
        counts   = Counter(lower)
        vocab    = len(counts)

        out[i, 1]  = n_tokens
        out[i, 2]  = lengths.mean()
        out[i, 3]  = lengths.std()
        out[i, 4]  = vocab / n_tokens
        out[i, 5]  = sum(c == 1 for c in counts.values()) / n_tokens
        out[i, 6]  = sum(c * (c - 1) for c in counts.values()) / max(1, n_tokens * (n_tokens - 1))
        out[i, 7]  = np.mean(lengths <= 3)
        out[i, 8]  = np.mean((lengths >= 4) & (lengths <= 6))
        out[i, 9]  = np.mean((lengths >= 7) & (lengths <= 9))
        out[i, 10] = np.mean(lengths >= 10)

    return out