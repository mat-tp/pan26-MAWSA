"""
Syntax and discourse features for sentence-level stylometry.

New feature group: "syntax"
Captures:
  - Clause-level complexity via subordination signals
  - Discourse connectives and hedging language
  - Sentence opening patterns (start word categories)
  - Negation patterns
  - Quotation and citation markers
  - Numeric token patterns
  - Question-word types
  - Imperative & passive voice proxies
"""

import re
from collections import Counter

import numpy as np

# ── Pattern lists ──────────────────────────────────────────────────────────────
_SUBORDINATORS = frozenset({
    "because", "although", "though", "unless", "until", "since", "while",
    "whereas", "if", "when", "before", "after", "as", "once", "provided",
    "supposing", "whether", "whenever", "wherever",
})
_DISCOURSE_CONNECTIVES = frozenset({
    "however", "therefore", "thus", "hence", "furthermore", "moreover",
    "additionally", "consequently", "nevertheless", "nonetheless", "meanwhile",
    "subsequently", "finally", "firstly", "secondly", "lastly", "additionally",
    "in contrast", "for example", "for instance", "in summary", "in conclusion",
})
_HEDGES = frozenset({
    "maybe", "perhaps", "probably", "possibly", "apparently", "seemingly",
    "roughly", "approximately", "arguably", "presumably", "supposedly",
    "generally", "usually", "often", "sometimes", "tend", "tends", "seemed",
    "appears", "appear", "seems", "suggest", "suggests", "indicate", "indicates",
    "might", "may", "could", "would", "should",
})
_NEGATIONS = frozenset({
    "not", "no", "never", "neither", "nor", "none", "nothing", "nobody",
    "nowhere", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "n't",
})
_BOOSTERS = frozenset({
    "very", "extremely", "absolutely", "completely", "totally", "definitely",
    "certainly", "clearly", "obviously", "undoubtedly", "quite", "really",
    "truly", "highly", "strongly", "deeply",
})
_QUESTION_WORDS = frozenset({"who", "what", "where", "when", "why", "how", "which", "whom", "whose"})
_PASSIVE_SIGNALS = frozenset({"was", "were", "is", "are", "been", "being", "be"})
_PASSIVE_PAST_RE = re.compile(r"\b(?:was|were|is|are|been|being|be)\s+\w+ed\b", re.IGNORECASE)

_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_TOKEN_RE = re.compile(r"[a-z]+", re.IGNORECASE)


NAMES = [
    # subordination / complexity
    "subordinator_ratio",
    "n_subordinators",
    "n_commas_per_word",       # comma count proxy for clause count
    "clause_depth_proxy",      # commas + semicolons per word

    # discourse
    "discourse_connective_ratio",
    "hedge_ratio",
    "booster_ratio",

    # negation
    "negation_ratio",

    # numerics
    "numeric_token_ratio",
    "has_percentage",
    "has_currency",

    # question / imperative / passive proxies
    "starts_with_question_word",
    "ends_with_question_mark",
    "passive_voice_proxy",

    # opening word category
    "starts_with_i",
    "starts_with_det",          # a / an / the
    "starts_with_conjunction",

    # quoting / citation
    "has_quote_marker",         # presence of " or '...'
    "has_url",

    # char-level entropy (captures script/Unicode diversity)
    "char_entropy",
]

_DETERMINERS_OPEN = frozenset({"a", "an", "the"})
_CONJ_OPEN = frozenset({"and", "but", "or", "so", "yet", "nor"})
_CURRENCY_RE = re.compile(r"[$£€¥₹]|\b(?:usd|eur|gbp|jpy)\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_QUOTE_RE = re.compile(r'["\u201c\u201d\u2018\u2019]|\'\'|``')


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    counts = Counter(text)
    probs = np.array([v / n for v in counts.values()], dtype=np.float64)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def extract(sentence: str) -> np.ndarray:
    tokens = _TOKEN_RE.findall(sentence.lower())
    n_tokens = max(1, len(tokens))
    token_set = set(tokens)
    first_token = tokens[0] if tokens else ""

    n_sub = sum(1 for t in tokens if t in _SUBORDINATORS)
    n_disc = sum(1 for t in tokens if t in _DISCOURSE_CONNECTIVES)
    n_hedge = sum(1 for t in tokens if t in _HEDGES)
    n_boost = sum(1 for t in tokens if t in _BOOSTERS)
    n_neg = sum(1 for t in tokens if t in _NEGATIONS)
    n_nums = len(_NUM_RE.findall(sentence))
    n_commas = sentence.count(",")
    n_semis = sentence.count(";")

    passive_count = len(_PASSIVE_PAST_RE.findall(sentence))

    return np.array([
        n_sub / n_tokens,
        float(n_sub),
        n_commas / n_tokens,
        (n_commas + n_semis) / n_tokens,

        n_disc / n_tokens,
        n_hedge / n_tokens,
        n_boost / n_tokens,

        n_neg / n_tokens,

        n_nums / n_tokens,
        float(bool(_CURRENCY_RE.search(sentence))),
        float("%" in sentence),

        float(first_token in _QUESTION_WORDS),
        float(sentence.rstrip().endswith("?")),
        passive_count / n_tokens,

        float(first_token == "i"),
        float(first_token in _DETERMINERS_OPEN),
        float(first_token in _CONJ_OPEN),

        float(bool(_QUOTE_RE.search(sentence))),
        float(bool(_URL_RE.search(sentence))),

        _entropy(sentence),
    ], dtype=np.float32)


def extract_batch(sentences) -> np.ndarray:
    n = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)
    for i, s in enumerate(sentences):
        out[i] = extract(s)
    return out
