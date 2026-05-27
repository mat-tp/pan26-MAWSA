"""
Function word features.

Function words (prepositions, conjunctions, determiners, pronouns, auxiliary
verbs) are topic-independent stylometric markers — they reveal HOW an author
writes rather than WHAT they write about.  This makes them especially
valuable for author identification tasks where topic similarity could
otherwise fool a classifier.

The word list here is drawn from standard English function word sets used in
stylometry research (cf. Mosteller & Wallace 1964; Koppel et al. 2009).
It intentionally excludes content words (nouns, main verbs, adjectives).
"""

import re

# ~150 genuine English function words grouped by category.
# These are closed-class words that carry grammatical rather than content meaning.
_FUNCTION_WORDS = frozenset([
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "each", "every", "either", "neither",
    "both", "all", "half", "several", "few", "many", "much",
    "more", "most", "other", "another", "such",
    # Personal pronouns
    "i", "me", "my", "myself",
    "you", "your", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    # Relative / interrogative pronouns
    "who", "whom", "whose", "which", "what",
    # Demonstrative pronouns (not repeated from determiners)
    "one", "ones",
    # Prepositions
    "in", "on", "at", "to", "for", "of", "from", "by",
    "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "under",
    "over", "out", "off", "up", "down", "near", "along",
    "around", "among", "within", "without", "beyond", "beside",
    "towards", "upon", "across", "behind", "beneath", "except",
    "per", "via", "versus",
    # Coordinating conjunctions
    "and", "but", "or", "nor", "so", "yet", "for",
    # Subordinating conjunctions
    "because", "although", "though", "while", "whereas",
    "since", "if", "unless", "until", "when", "whenever",
    "where", "wherever", "whether", "as", "than", "that",
    "once", "before", "after", "since",
    # Auxiliary / modal verbs
    "be", "is", "are", "was", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did",
    "will", "would", "shall", "should",
    "can", "could", "may", "might", "must",
    "ought", "need", "dare",
    # Negation
    "not", "no",
    # Quantifiers
    "enough", "quite", "rather", "too", "very", "just",
    "only", "also", "even", "already", "still", "yet",
    "again", "once", "twice",
    # Discourse markers / conjunctive adverbs
    "however", "therefore", "thus", "hence", "moreover",
    "furthermore", "nevertheless", "nonetheless", "otherwise",
    "meanwhile", "indeed", "instead", "then", "now", "here",
    "there", "so",
])

# Per-word features: one frequency per function word (sparse but interpretable)
# We only use the top-N most common in practice (configured in pipeline)
INDIVIDUAL_WORDS = sorted(_FUNCTION_WORDS)

NAMES = [
    "fw_count",     # absolute count of function words
    "fw_ratio",     # function words / total tokens
    "stopword_ratio",  # same as fw_ratio but named for clarity in reports
]


def extract(sentence):
    """Return aggregate function word features for one sentence."""
    tokens = re.findall(r"[a-z]+", sentence.lower())
    n_tokens = max(1, len(tokens))

    fw_count = sum(1 for t in tokens if t in _FUNCTION_WORDS)
    fw_ratio = fw_count / n_tokens

    return [fw_count, fw_ratio, fw_ratio]  # stopword_ratio == fw_ratio


def extract_per_word(sentence, word_list=None):
    """
    Return a frequency vector over individual function words.

    word_list defaults to INDIVIDUAL_WORDS (all ~150 words).
    Pass a shorter list for a more compact representation.

    Returns a list of floats (normalised counts) in the same order as word_list.
    """
    if word_list is None:
        word_list = INDIVIDUAL_WORDS

    tokens = re.findall(r"[a-z]+", sentence.lower())
    n_tokens = max(1, len(tokens))
    counts = {}
    for t in tokens:
        if t in _FUNCTION_WORDS:
            counts[t] = counts.get(t, 0) + 1

    return [counts.get(w, 0) / n_tokens for w in word_list]


def get_names_per_word(word_list=None):
    """Return feature names for the per-word vector."""
    if word_list is None:
        word_list = INDIVIDUAL_WORDS
    return [f"fw_{w}" for w in word_list]
