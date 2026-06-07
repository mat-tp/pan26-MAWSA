"""
Punctuation and surface-form features.
"""

from collections import Counter

import numpy as np

NAMES = [
    "n_commas",
    "n_periods",
    "n_semicolons",
    "n_colons",
    "n_exclamations",
    "n_questions",
    "n_quotes",
    "n_parens",
    "n_dashes",
    "n_ellipses",
    "punct_density",
    "digit_ratio",
    "upper_ratio",
    "comma_period_ratio",
]


def extract(sentence):
    """Extract punctuation features for a single sentence."""
    n_chars  = max(1, len(sentence))
    n_words  = max(1, len(sentence.split()))
    counts   = Counter(sentence)

    n_commas     = counts[","]
    n_periods    = counts["."]
    n_semicolons = counts[";"]
    n_colons     = counts[":"]
    n_excl       = counts["!"]
    n_quest      = counts["?"]
    n_quotes     = counts['"'] + counts["'"]
    n_parens     = counts["("] + counts[")"]
    n_dashes     = counts["-"] + counts["—"]
    n_ellipses   = sentence.count("...")
    punct_total  = sum(1 for c in sentence if not c.isalnum() and not c.isspace())

    return np.array([
        n_commas, n_periods, n_semicolons, n_colons,
        n_excl, n_quest, n_quotes, n_parens, n_dashes, n_ellipses,
        punct_total / n_words,
        sum(c.isdigit() for c in sentence) / n_chars,
        sum(c.isupper() for c in sentence) / n_chars,
        n_commas / max(1, n_periods),
    ], dtype=np.float32)


def extract_batch(sentences):
    """Vectorised batch extraction over a list of sentences."""
    n   = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        n_chars  = max(1, len(sentence))
        n_words  = max(1, len(sentence.split()))
        counts   = Counter(sentence)

        n_commas     = counts[","]
        n_periods    = counts["."]
        n_semicolons = counts[";"]
        n_colons     = counts[":"]
        n_excl       = counts["!"]
        n_quest      = counts["?"]
        n_quotes     = counts['"'] + counts["'"]
        n_parens     = counts["("] + counts[")"]
        n_dashes     = counts["-"] + counts["—"]
        n_ellipses   = sentence.count("...")
        punct_total  = sum(1 for c in sentence if not c.isalnum() and not c.isspace())

        out[i, 0]  = n_commas
        out[i, 1]  = n_periods
        out[i, 2]  = n_semicolons
        out[i, 3]  = n_colons
        out[i, 4]  = n_excl
        out[i, 5]  = n_quest
        out[i, 6]  = n_quotes
        out[i, 7]  = n_parens
        out[i, 8]  = n_dashes
        out[i, 9]  = n_ellipses
        out[i, 10] = punct_total / n_words
        out[i, 11] = sum(c.isdigit() for c in sentence) / n_chars
        out[i, 12] = sum(c.isupper() for c in sentence) / n_chars
        out[i, 13] = n_commas / max(1, n_periods)

    return out