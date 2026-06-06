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
    """
    Vectorised batch extraction.

    Processes all sentences together using numpy operations where possible,
    avoiding a Python-level loop over sentences for the hot inner path.
    """
    n = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        words = _TOKEN_RE.findall(sentence)

        out[i, 0] = len(sentence)          # n_chars

        if not words:
            out[i, 1] = 1                  # n_tokens minimum
            continue

        n_tokens   = len(words)
        lengths    = np.array([len(w) for w in words], dtype=np.float32)
        lower      = [w.lower() for w in words]
        counts     = Counter(lower)
        vocab_size = len(counts)

        out[i, 1]  = n_tokens
        out[i, 2]  = lengths.mean()
        out[i, 3]  = lengths.std()
        out[i, 4]  = vocab_size / n_tokens                              # ttr
        out[i, 5]  = sum(c == 1 for c in counts.values()) / n_tokens   # hapax
        out[i, 6]  = (
            sum(c * (c - 1) for c in counts.values())
            / max(1, n_tokens * (n_tokens - 1))
        )                                                                # simpson_d
        out[i, 7]  = np.mean(lengths <= 3)
        out[i, 8]  = np.mean((lengths >= 4) & (lengths <= 6))
        out[i, 9]  = np.mean((lengths >= 7) & (lengths <= 9))
        out[i, 10] = np.mean(lengths >= 10)

    return out