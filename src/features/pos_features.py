"""
POS stylometric features.

Batch POS tagging via nltk.pos_tag_sents() with a per-sentence tag cache
to avoid redundant work across multiple extract_batch() calls.
"""

from collections import Counter

import numpy as np

UNIVERSAL_TAGS = ["ADJ", "ADP", "ADV", "CONJ", "DET", "NOUN", "NUM", "PRT", "PRON", "VERB", ".", "X"]

POS_BIGRAMS_TRACKED = [
    ("DET", "NOUN"), ("ADJ", "NOUN"), ("NOUN", "VERB"), ("VERB", "NOUN"),
    ("VERB", "ADV"),  ("ADV", "ADJ"),  ("PRON", "VERB"), ("VERB", "ADP"),
    ("ADP", "NOUN"),  ("CONJ", "NOUN"), ("CONJ", "VERB"), ("DET", "ADJ"),
]

NAMES_TAG_DIST = [f"pos_{t.lower()}" for t in UNIVERSAL_TAGS]
NAMES_BIGRAMS  = [f"posbg_{a.lower()}_{b.lower()}" for a, b in POS_BIGRAMS_TRACKED]
NAMES_EXTRA    = ["noun_verb_ratio", "adj_noun_ratio", "pronoun_noun_ratio"]
NAMES          = NAMES_TAG_DIST + NAMES_BIGRAMS + NAMES_EXTRA

# In-memory LRU-style cache; cleared when it reaches the cap
_TAG_CACHE     = {}
_TAG_CACHE_MAX = 100_000
POS_BATCH_SIZE = 5_000


def ensure_nltk_data():
    """Download only the resources required by this module."""
    import nltk
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def _tag(sentence):
    """Cached single-sentence POS tagging."""
    if sentence in _TAG_CACHE:
        return _TAG_CACHE[sentence]

    import nltk
    from nltk.tokenize import wordpunct_tokenize

    tokens = wordpunct_tokenize(sentence)
    tagged = nltk.pos_tag(tokens, tagset="universal")
    tags   = tuple(tag for _, tag in tagged)

    if len(_TAG_CACHE) >= _TAG_CACHE_MAX:
        _TAG_CACHE.clear()
    _TAG_CACHE[sentence] = tags
    return tags


def extract(sentence):
    """Extract POS features for a single sentence."""
    try:
        tags = list(_tag(sentence))
    except Exception:
        tags = []
    return _tags_to_features(tags)


def extract_batch(sentences):
    """
    Fast batch POS extraction.

    Strategy: serve cached sentences immediately; collect uncached ones and
    tag them in chunks via pos_tag_sents() for efficiency.

    Returns ndarray of shape (n_sentences, len(NAMES)).
    """
    import nltk
    from nltk.tokenize import wordpunct_tokenize

    ensure_nltk_data()

    n_sentences = len(sentences)
    if n_sentences == 0:
        return np.empty((0, len(NAMES)), dtype=np.float32)

    tag_results    = [None] * n_sentences
    fresh_idx      = []
    fresh_sents    = []

    for i, sentence in enumerate(sentences):
        cached = _TAG_CACHE.get(sentence)
        if cached is not None:
            tag_results[i] = list(cached)
        else:
            fresh_idx.append(i)
            fresh_sents.append(sentence)

    if fresh_sents:
        print(f"[pos] Tagging {len(fresh_sents):,} uncached sentences...")
        for start in range(0, len(fresh_sents), POS_BATCH_SIZE):
            end            = min(start + POS_BATCH_SIZE, len(fresh_sents))
            batch_tokens   = [wordpunct_tokenize(s) for s in fresh_sents[start:end]]
            batch_tagged   = nltk.pos_tag_sents(batch_tokens, tagset="universal")

            for pos_idx, sentence, tagged in zip(fresh_idx[start:end], fresh_sents[start:end], batch_tagged):
                tags = tuple(tag for _, tag in tagged)
                if len(_TAG_CACHE) >= _TAG_CACHE_MAX:
                    _TAG_CACHE.clear()
                _TAG_CACHE[sentence]   = tags
                tag_results[pos_idx]   = list(tags)

            if start % 50_000 == 0:
                print(f"[pos] Processed {end:,}/{len(fresh_sents):,}")

    return np.vstack([_tags_to_features(tags or []) for tags in tag_results]).astype(np.float32)


def _tags_to_features(tags):
    """Convert a POS tag sequence into stylometric feature vector."""
    if not tags:
        return np.zeros(len(NAMES), dtype=np.float32)

    n          = max(1, len(tags))
    tag_counts = Counter(tags)

    tag_dist     = [tag_counts.get(tag, 0) / n for tag in UNIVERSAL_TAGS]
    bigram_cnt   = Counter(zip(tags[:-1], tags[1:]))
    n_bigrams    = max(1, len(tags) - 1)
    bigram_feats = [bigram_cnt.get(pair, 0) / n_bigrams for pair in POS_BIGRAMS_TRACKED]

    nouns = tag_counts.get("NOUN", 0)
    verbs = tag_counts.get("VERB", 0)
    adjs  = tag_counts.get("ADJ",  0)
    prons = tag_counts.get("PRON", 0)
    extra = [nouns / max(1, verbs), adjs / max(1, nouns), prons / max(1, nouns)]

    return np.array(tag_dist + bigram_feats + extra, dtype=np.float32)