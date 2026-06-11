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
    "contraction_ratio",
    "capitalized_word_ratio",
    "all_caps_word_ratio",
    "elongated_word_ratio",
]


def _word_tokens(sentence):
    return [token for token in sentence.split() if any(ch.isalpha() for ch in token)]


def _is_contraction(token):
    return "'" in token and len(token) > 2


def _is_all_caps(token):
    letters = [c for c in token if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _is_capitalized(token):
    return token[:1].isupper() and token[1:].islower()


def _is_elongated(token):
    return any(ch * 3 in token.lower() for ch in set(token.lower()))


def extract(sentence):
    """Extract punctuation and surface features for a single sentence."""
    n_chars  = max(1, len(sentence))
    tokens   = _word_tokens(sentence)
    n_words  = max(1, len(tokens))
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
    contractions = sum(1 for token in tokens if _is_contraction(token))
    cap_words = sum(1 for token in tokens if _is_capitalized(token))
    all_caps = sum(1 for token in tokens if _is_all_caps(token))
    elongated = sum(1 for token in tokens if _is_elongated(token))

    return np.array([
        n_commas, n_periods, n_semicolons, n_colons,
        n_excl, n_quest, n_quotes, n_parens, n_dashes, n_ellipses,
        punct_total / n_words,
        sum(c.isdigit() for c in sentence) / n_chars,
        sum(c.isupper() for c in sentence) / n_chars,
        n_commas / max(1, n_periods),
        contractions / n_words,
        cap_words / n_words,
        all_caps / n_words,
        elongated / n_words,
    ], dtype=np.float32)


def extract_batch(sentences):
    """Vectorised batch extraction over a list of sentences."""
    n   = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)

    for i, sentence in enumerate(sentences):
        n_chars  = max(1, len(sentence))
        tokens   = _word_tokens(sentence)
        n_words  = max(1, len(tokens))
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
        contractions = sum(1 for token in tokens if _is_contraction(token))
        cap_words    = sum(1 for token in tokens if _is_capitalized(token))
        all_caps     = sum(1 for token in tokens if _is_all_caps(token))
        elongated    = sum(1 for token in tokens if _is_elongated(token))

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
        out[i, 14] = contractions / n_words
        out[i, 15] = cap_words / n_words
        out[i, 16] = all_caps / n_words
        out[i, 17] = elongated / n_words

    return out