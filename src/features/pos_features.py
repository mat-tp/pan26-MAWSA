"""
POS stylometric features.

Optimized for large-scale extraction:
    - Batch POS tagging via nltk.pos_tag_sents()
    - Sentence-level tag caching
    - Chunked processing for memory efficiency
    - No dependency on punkt/punkt_tab
"""

from collections import Counter

import numpy as np

UNIVERSAL_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "CONJ",
    "DET",
    "NOUN",
    "NUM",
    "PRT",
    "PRON",
    "VERB",
    ".",
    "X",
]

POS_BIGRAMS_TRACKED = [
    ("DET", "NOUN"),
    ("ADJ", "NOUN"),
    ("NOUN", "VERB"),
    ("VERB", "NOUN"),
    ("VERB", "ADV"),
    ("ADV", "ADJ"),
    ("PRON", "VERB"),
    ("VERB", "ADP"),
    ("ADP", "NOUN"),
    ("CONJ", "NOUN"),
    ("CONJ", "VERB"),
    ("DET", "ADJ"),
]

NAMES_TAG_DIST = [
    f"pos_{t.lower()}"
    for t in UNIVERSAL_TAGS
]

NAMES_BIGRAMS = [
    f"posbg_{a.lower()}_{b.lower()}"
    for a, b in POS_BIGRAMS_TRACKED
]

NAMES_EXTRA = [
    "noun_verb_ratio",
    "adj_noun_ratio",
    "pronoun_noun_ratio",
]

NAMES = (
    NAMES_TAG_DIST
    + NAMES_BIGRAMS
    + NAMES_EXTRA
)

# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------

_TAG_CACHE = {}

_TAG_CACHE_MAX = 100_000

POS_BATCH_SIZE = 5000


# ------------------------------------------------------------------
# NLTK
# ------------------------------------------------------------------

def ensure_nltk_data():
    """
    Download only resources actually required by this module.
    """
    import nltk

    resources = {
        "taggers/averaged_perceptron_tagger_eng":
            "averaged_perceptron_tagger_eng",
    }

    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


# ------------------------------------------------------------------
# Tagging
# ------------------------------------------------------------------

def _tag(sentence):
    """
    Cached single-sentence tagging.
    """

    if sentence in _TAG_CACHE:
        return _TAG_CACHE[sentence]

    import nltk
    from nltk.tokenize import wordpunct_tokenize

    tokens = wordpunct_tokenize(sentence)

    tagged = nltk.pos_tag(
        tokens,
        tagset="universal",
    )

    tags = tuple(
        tag
        for _, tag in tagged
    )

    if len(_TAG_CACHE) >= _TAG_CACHE_MAX:
        _TAG_CACHE.clear()

    _TAG_CACHE[sentence] = tags

    return tags


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def extract(sentence):
    """
    Extract POS features for a single sentence.
    """

    try:
        tags = list(_tag(sentence))
    except Exception:
        tags = []

    return _tags_to_features(tags)


def extract_batch(sentences):
    """
    Fast batch POS extraction.

    Strategy:
        1. Reuse cached sentences.
        2. Collect uncached sentences.
        3. Tag them in chunks using pos_tag_sents().
        4. Convert all tag sequences to stylometric features.

    Returns:
        np.ndarray shape:
            (n_sentences, len(NAMES))
    """

    import nltk
    from nltk.tokenize import wordpunct_tokenize

    ensure_nltk_data()

    n_sentences = len(sentences)

    if n_sentences == 0:
        return np.empty(
            (0, len(NAMES)),
            dtype=np.float32,
        )

    tag_results = [None] * n_sentences

    fresh_positions = []
    fresh_sentences = []

    # ----------------------------------------------------------
    # Cache lookup
    # ----------------------------------------------------------

    for i, sentence in enumerate(sentences):

        cached = _TAG_CACHE.get(sentence)

        if cached is not None:
            tag_results[i] = list(cached)
        else:
            fresh_positions.append(i)
            fresh_sentences.append(sentence)

    # ----------------------------------------------------------
    # Batch tagging
    # ----------------------------------------------------------

    if fresh_sentences:

        print(
            f"[pos] Tagging "
            f"{len(fresh_sentences):,} "
            f"uncached sentences..."
        )

        for start in range(
            0,
            len(fresh_sentences),
            POS_BATCH_SIZE,
        ):

            end = min(
                start + POS_BATCH_SIZE,
                len(fresh_sentences),
            )

            batch_sentences = fresh_sentences[start:end]
            batch_positions = fresh_positions[start:end]

            batch_tokens = [
                wordpunct_tokenize(s)
                for s in batch_sentences
            ]

            batch_tagged = nltk.pos_tag_sents(
                batch_tokens,
                tagset="universal",
            )

            for pos_idx, sentence, tagged in zip(
                batch_positions,
                batch_sentences,
                batch_tagged,
            ):

                tags = tuple(
                    tag
                    for _, tag in tagged
                )

                if len(_TAG_CACHE) >= _TAG_CACHE_MAX:
                    _TAG_CACHE.clear()

                _TAG_CACHE[sentence] = tags

                tag_results[pos_idx] = list(tags)

            if start % 50000 == 0:
                print(
                    f"[pos] Processed "
                    f"{end:,}/{len(fresh_sentences):,}"
                )

    # ----------------------------------------------------------
    # Convert tags → features
    # ----------------------------------------------------------

    return np.vstack(
        [
            _tags_to_features(tags or [])
            for tags in tag_results
        ]
    ).astype(np.float32)


# ------------------------------------------------------------------
# Feature construction
# ------------------------------------------------------------------

def _tags_to_features(tags):
    """
    Convert POS tag sequence into stylometric features.
    """

    if not tags:
        return np.zeros(
            len(NAMES),
            dtype=np.float32,
        )

    n = max(1, len(tags))

    tag_counts = Counter(tags)

    tag_dist = [
        tag_counts.get(tag, 0) / n
        for tag in UNIVERSAL_TAGS
    ]

    bigram_counts = Counter(
        zip(tags[:-1], tags[1:])
    )

    n_bigrams = max(
        1,
        len(tags) - 1,
    )

    bigram_feats = [
        bigram_counts.get(pair, 0)
        / n_bigrams
        for pair in POS_BIGRAMS_TRACKED
    ]

    nouns = tag_counts.get("NOUN", 0)
    verbs = tag_counts.get("VERB", 0)
    adjs = tag_counts.get("ADJ", 0)
    prons = tag_counts.get("PRON", 0)

    extra = [
        nouns / max(1, verbs),
        adjs / max(1, nouns),
        prons / max(1, nouns),
    ]

    return np.array(
        tag_dist
        + bigram_feats
        + extra,
        dtype=np.float32,
    )