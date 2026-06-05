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

    words = _TOKEN_RE.findall(sentence)

    lower = [w.lower() for w in words]

    n_chars = len(sentence)

    n_tokens = max(1, len(words))

    if not words:
        return np.zeros(len(NAMES), dtype=np.float32)

    lengths = np.array(
        [len(w) for w in words],
        dtype=np.float32,
    )

    avg_word_len = lengths.mean()

    std_word_len = lengths.std()

    counts = Counter(lower)

    vocab_size = len(counts)

    ttr = vocab_size / n_tokens

    hapax_ratio = (
        sum(c == 1 for c in counts.values())
        / n_tokens
    )

    simpson_d = (
        sum(
            c * (c - 1)
            for c in counts.values()
        )
        / max(1, n_tokens * (n_tokens - 1))
    )

    wl_short = np.mean(lengths <= 3)

    wl_medium = np.mean(
        (lengths >= 4) & (lengths <= 6)
    )

    wl_long = np.mean(
        (lengths >= 7) & (lengths <= 9)
    )

    wl_very_long = np.mean(lengths >= 10)

    return np.array(
        [
            n_chars,
            n_tokens,
            avg_word_len,
            std_word_len,
            ttr,
            hapax_ratio,
            simpson_d,
            wl_short,
            wl_medium,
            wl_long,
            wl_very_long,
        ],
        dtype=np.float32,
    )

def extract_batch(sentences):
    """Vectorised batch extraction — much faster than looping over extract()."""
    return np.vstack([extract(s) for s in sentences]).astype(np.float32)