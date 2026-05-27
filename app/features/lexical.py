"""
Lexical features: sentence length, word length, vocabulary richness.

All functions take a single sentence string and return a flat list of floats.
Feature names are listed in NAMES so the pipeline can label them correctly.
"""

import re
from collections import Counter

NAMES = [
    "n_chars",
    "n_tokens",
    "avg_word_len",
    "ttr",           # type-token ratio
    "hapax_ratio",   # fraction of words appearing exactly once
    "simpson_d",     # Simpson's diversity index (vocabulary concentration)
    "wl_short",      # fraction of words with length <= 3
    "wl_medium",     # fraction of words with length 4-6
    "wl_long",       # fraction of words with length 7-9
    "wl_very_long",  # fraction of words with length >= 10
]


def extract(sentence):
    """Return lexical feature values for one sentence."""
    alpha = re.findall(r"[a-zA-Z]+", sentence)
    lower = [w.lower() for w in alpha]

    n_chars = len(sentence)
    n_tokens = max(1, len(sentence.split()))
    n_alpha = max(1, len(alpha))

    avg_word_len = sum(len(w) for w in alpha) / n_alpha

    # Vocabulary richness — computed on lowercased alpha tokens
    if not lower:
        ttr, hapax_ratio, simpson_d = 0.0, 0.0, 0.0
    else:
        counts = Counter(lower)
        n = len(lower)
        ttr = len(counts) / n
        hapax_ratio = sum(1 for c in counts.values() if c == 1) / n
        # Simpson's D: probability that two random tokens are the same type
        # Low value = diverse vocabulary; high value = repetitive
        simpson_d = sum(c * (c - 1) for c in counts.values()) / max(1, n * (n - 1))

    # Word-length bins (normalised by n_alpha)
    bins = [0, 0, 0, 0]
    for w in alpha:
        wl = len(w)
        if wl <= 3:
            bins[0] += 1
        elif wl <= 6:
            bins[1] += 1
        elif wl <= 9:
            bins[2] += 1
        else:
            bins[3] += 1
    wl_short, wl_medium, wl_long, wl_very_long = [b / n_alpha for b in bins]

    return [
        n_chars, n_tokens, avg_word_len,
        ttr, hapax_ratio, simpson_d,
        wl_short, wl_medium, wl_long, wl_very_long,
    ]
