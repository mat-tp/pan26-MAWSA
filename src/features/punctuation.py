"""
Punctuation features.
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

PUNCT_SET = set(",.;:!?\"'()-[]{}")


def extract(sentence):

    n_chars = max(1, len(sentence))

    n_words = max(1, len(sentence.split()))

    counts = Counter(sentence)

    n_commas = counts[","]

    n_periods = counts["."]

    n_semicolons = counts[";"]

    n_colons = counts[":"]

    n_excl = counts["!"]

    n_quest = counts["?"]

    n_quotes = counts['"'] + counts["'"]

    n_parens = (
        counts["("]
        + counts[")"]
    )

    n_dashes = (
        counts["-"]
        + counts["—"]
    )

    n_ellipses = sentence.count("...")

    punct_total = sum(
        1
        for c in sentence
        if not c.isalnum()
        and not c.isspace()
    )

    punct_density = punct_total / n_words

    digit_ratio = (
        sum(c.isdigit() for c in sentence)
        / n_chars
    )

    upper_ratio = (
        sum(c.isupper() for c in sentence)
        / n_chars
    )

    comma_period_ratio = (
        n_commas / max(1, n_periods)
    )

    return np.array(
        [
            n_commas,
            n_periods,
            n_semicolons,
            n_colons,
            n_excl,
            n_quest,
            n_quotes,
            n_parens,
            n_dashes,
            n_ellipses,
            punct_density,
            digit_ratio,
            upper_ratio,
            comma_period_ratio,
        ],
        dtype=np.float32,
    )

def extract_batch(sentences):
    """Vectorised batch extraction."""
    return np.vstack([extract(s) for s in sentences]).astype(np.float32)