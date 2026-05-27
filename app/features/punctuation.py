"""
Punctuation features: counts and density of punctuation marks.

Punctuation usage is a reliable stylometric signal — different authors
tend to favour different punctuation habits consistently across their writing.
"""

import string

NAMES = [
    "n_commas",
    "n_periods",
    "n_semicolons",
    "n_colons",
    "n_exclamations",
    "n_questions",
    "n_quotes",       # both " and '
    "n_parens",       # ( and )
    "n_dashes",       # - and —
    "n_ellipses",     # ...
    "punct_density",  # total punctuation / n_words
    "digit_ratio",    # digits / n_chars
    "upper_ratio",    # uppercase letters / n_chars
]


def extract(sentence):
    """Return punctuation feature values for one sentence."""
    n_words = max(1, len(sentence.split()))
    n_chars = max(1, len(sentence))

    n_commas      = sentence.count(",")
    n_periods     = sentence.count(".")
    n_semicolons  = sentence.count(";")
    n_colons      = sentence.count(":")
    n_excl        = sentence.count("!")
    n_quest       = sentence.count("?")
    n_quotes      = sentence.count('"') + sentence.count("'")
    n_parens      = sentence.count("(") + sentence.count(")")
    n_dashes      = sentence.count("-") + sentence.count("—")
    n_ellipses    = sentence.count("...")

    n_punct       = sum(1 for c in sentence if c in string.punctuation)
    punct_density = n_punct / n_words

    n_digits      = sum(c.isdigit() for c in sentence)
    digit_ratio   = n_digits / n_chars

    n_upper       = sum(c.isupper() for c in sentence)
    upper_ratio   = n_upper / n_chars

    return [
        n_commas, n_periods, n_semicolons, n_colons,
        n_excl, n_quest, n_quotes, n_parens, n_dashes, n_ellipses,
        punct_density, digit_ratio, upper_ratio,
    ]
