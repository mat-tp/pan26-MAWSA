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
    "honore_r",
    "brunet_w",
    "yules_k",
    "mattr",
    "wl_short",
    "wl_medium",
    "wl_long",
    "wl_very_long",
]


def _count_syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    syllables = re.findall(r"[aeiouy]{1,2}", word)
    return max(1, len(syllables))


def _mattr(words, window=100):
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n
    unique_counts = []
    for i in range(n - window + 1):
        chunk = words[i:i + window]
        unique_counts.append(len(set(chunk)) / window)
    return float(np.mean(unique_counts))


def _yules_k(counts, n_tokens):
    if n_tokens <= 1:
        return 0.0
    m1 = n_tokens
    m2 = sum(freq * freq for freq in counts.values())
    return 10000.0 * (m2 - m1) / (m1 * m1)


def _honore_r(vocab, n1, n_tokens):
    if vocab == 0 or n1 == 0:
        return 0.0
    denom = 1.0 - n1 / vocab
    if denom <= 1e-6:
        denom = 1e-6
    return 100.0 * np.log(n_tokens) / denom


def _brunet_w(vocab, n_tokens):
    if vocab <= 1 or n_tokens <= 1:
        return 0.0
    return 1000.0 * (n_tokens ** (-0.165))


def _lexical_stats(words):
    n_tokens = len(words)
    lengths = np.array([len(w) for w in words], dtype=np.float32)
    lower = [w.lower() for w in words]
    counts = Counter(lower)
    vocab = len(counts)
    n1 = sum(1 for c in counts.values() if c == 1)

    return {
        "n_tokens": n_tokens,
        "lengths": lengths,
        "counts": counts,
        "vocab": vocab,
        "n1": n1,
    }


def _build_lexical_vector(stats, sentence_length, word_list):
    n_tokens = stats["n_tokens"]
    lengths = stats["lengths"]
    counts = stats["counts"]
    vocab = stats["vocab"]
    n1 = stats["n1"]

    return np.array([
        sentence_length,
        n_tokens,
        lengths.mean() if n_tokens else 0.0,
        lengths.std() if n_tokens else 0.0,
        vocab / n_tokens if n_tokens else 0.0,
        sum(c == 1 for c in counts.values()) / n_tokens if n_tokens else 0.0,
        sum(c * (c - 1) for c in counts.values()) / max(1, n_tokens * (n_tokens - 1)) if n_tokens > 1 else 0.0,
        _honore_r(vocab, n1, n_tokens),
        _brunet_w(vocab, n_tokens),
        _yules_k(counts, n_tokens),
        _mattr(word_list, window=min(100, n_tokens)),
        np.mean(lengths <= 3) if n_tokens else 0.0,
        np.mean((lengths >= 4) & (lengths <= 6)) if n_tokens else 0.0,
        np.mean((lengths >= 7) & (lengths <= 9)) if n_tokens else 0.0,
        np.mean(lengths >= 10) if n_tokens else 0.0,
    ], dtype=np.float32)


def extract(sentence):
    """Extract lexical features for a single sentence."""
    words = _TOKEN_RE.findall(sentence)
    if not words:
        return np.zeros(len(NAMES), dtype=np.float32)

    stats = _lexical_stats(words)
    return _build_lexical_vector(stats, len(sentence), [w.lower() for w in words])


def extract_batch(sentences):
    """Vectorised batch extraction over a list of sentences."""
    n   = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        words = _TOKEN_RE.findall(sentence)
        if not words:
            out[i, 0] = len(sentence)
            out[i, 1] = 1
            continue

        stats = _lexical_stats(words)
        out[i] = _build_lexical_vector(stats, len(sentence), [w.lower() for w in words])

    return out