"""
POS-based features using NLTK's universal POS tagset.

POS distributions capture syntactic style — how much an author relies on
nouns vs verbs vs adjectives, their preference for subordinate clauses, etc.
These features are topic-independent because they describe sentence structure,
not vocabulary.

We use NLTK's averaged perceptron tagger with the universal tagset (12 tags),
which keeps the feature space small and interpretable.

Universal tags: ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRT, PRON, VERB, ., X

First run: call ensure_nltk_data() to download required NLTK models.
"""

import re

import numpy as np

# Universal POS tagset — 12 coarse tags (plus sentence boundary)
UNIVERSAL_TAGS = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN",
                  "NUM", "PRT", "PRON", "VERB", ".", "X"]

# POS bigrams we track (subject + verb, verb + object patterns)
# We focus on the most interpretable combinations rather than all O(144) pairs.
POS_BIGRAMS_TRACKED = [
    ("DET",  "NOUN"),   # determiner → noun (nominal phrases)
    ("ADJ",  "NOUN"),   # adjective modifier
    ("NOUN", "VERB"),   # subject → verb
    ("VERB", "NOUN"),   # verb → object
    ("VERB", "ADV"),    # verb → adverb
    ("ADV",  "ADJ"),    # adverb modifier
    ("PRON", "VERB"),   # pronoun subject
    ("VERB", "ADP"),    # verb → preposition
    ("ADP",  "NOUN"),   # preposition → noun phrase
    ("CONJ", "NOUN"),   # conjunction → noun
    ("CONJ", "VERB"),   # conjunction → verb
    ("DET",  "ADJ"),    # determiner → adjective
]

NAMES_TAG_DIST = [f"pos_{t.lower()}" for t in UNIVERSAL_TAGS]
NAMES_BIGRAMS  = [f"posbg_{a.lower()}_{b.lower()}" for a, b in POS_BIGRAMS_TRACKED]
NAMES = NAMES_TAG_DIST + NAMES_BIGRAMS


def ensure_nltk_data():
    """Download required NLTK data if not already present."""
    import nltk
    for resource in ["averaged_perceptron_tagger_eng", "universal_tagset"]:
        try:
            nltk.data.find(f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _tag(sentence):
    """Tokenise and POS-tag a sentence. Returns list of (word, universal_tag)."""
    import nltk
    tokens = nltk.word_tokenize(sentence)
    return nltk.pos_tag(tokens, tagset="universal")


def extract(sentence):
    """
    Return POS feature values for one sentence.

    Returns a flat list of floats: tag distribution + bigram frequencies.
    Falls back to zeros if NLTK is unavailable.
    """
    try:
        tagged = _tag(sentence)
    except Exception:
        return [0.0] * len(NAMES)

    n = max(1, len(tagged))
    tags = [tag for _, tag in tagged]

    # Tag distribution (normalised)
    tag_dist = [tags.count(t) / n for t in UNIVERSAL_TAGS]

    # POS bigram frequencies (normalised)
    n_bigrams = max(1, len(tags) - 1)
    bigram_counts = {}
    for a, b in zip(tags[:-1], tags[1:]):
        bigram_counts[(a, b)] = bigram_counts.get((a, b), 0) + 1

    bigram_feats = [bigram_counts.get(pair, 0) / n_bigrams
                    for pair in POS_BIGRAMS_TRACKED]

    return tag_dist + bigram_feats


def extract_batch(sentences):
    """Extract POS features for a list of sentences (reuses tagger)."""
    return np.array([extract(s) for s in sentences], dtype=np.float32)
