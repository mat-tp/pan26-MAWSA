"""
Readability and complexity features for sentence-level stylometry.
"""

import re

import numpy as np

NAMES = [
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "smog_index",
    "avg_syllables_per_word",
    "avg_chars_per_word",
]

_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]+")
_SENTENCE_END = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
    word = re.sub(r"^y", "", word)
    matches = _VOWEL_RE.findall(word)
    return max(1, len(matches))


def _polysyllable_count(words):
    return sum(1 for word in words if _count_syllables(word) >= 3)


def _word_tokens(sentence):
    return [token for token in re.findall(r"\b[\w'-]+\b", sentence) if re.search(r"[A-Za-z]", token)]


def extract(sentence):
    tokens = _word_tokens(sentence)
    n_words = max(1, len(tokens))
    syllables = sum(_count_syllables(w) for w in tokens)
    avg_syllables_per_word = syllables / n_words
    chars = sum(len(w) for w in tokens)
    avg_chars_per_word = chars / n_words

    # Sentence-level readability uses simple approximations.
    num_sentences = max(1, len(_SENTENCE_END.findall(sentence)) or 1)
    words_per_sentence = n_words / num_sentences
    score = 206.835 - 1.015 * words_per_sentence - 84.6 * avg_syllables_per_word
    grade = 0.39 * words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
    smog = 1.0430 * np.sqrt(_polysyllable_count(tokens) * 30.0 / max(1.0, num_sentences)) + 3.1291

    return np.array([
        score,
        grade,
        smog,
        avg_syllables_per_word,
        avg_chars_per_word,
    ], dtype=np.float32)


def extract_batch(sentences):
    n = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        out[i] = extract(sentence)
    return out
