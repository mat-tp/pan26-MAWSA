"""
Social-media and informal writing features for sentence-level stylometry.

New feature group: "social_media"
Designed for Twitter/social-media PAN tasks but also useful for general
informal corpora. Complements lexical/punctuation features with platform-
specific signals.
"""

import re
from collections import Counter

import numpy as np

_HASHTAG_RE   = re.compile(r"#\w+")
_MENTION_RE   = re.compile(r"@\w+")
_URL_RE       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMOJI_RE     = re.compile("["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
_KAOMOJI_RE   = re.compile(r"(?:[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8])")
_REPEATED_RE  = re.compile(r"(.)\1{2,}")          # repeated chars (looool, haaaa)
_ALL_CAPS_RE  = re.compile(r"\b[A-Z]{2,}\b")      # ALL CAPS words
_TOKEN_RE     = re.compile(r"\S+")

# Internet slang / filler words
_SLANG = frozenset({
    "lol", "lmao", "lmfao", "rofl", "omg", "wtf", "smh", "tbh",
    "imo", "imho", "idk", "ikr", "irl", "fyi", "tbt", "gg", "af",
    "ngl", "brb", "bff", "dm", "rt", "tfw", "til", "iirc", "afaik",
    "haha", "hehe", "hihi", "lmk", "asap", "btw", "thx", "ty", "np",
    "yolo", "fomo", "wfh",
})

NAMES = [
    # social structure
    "n_hashtags",
    "n_mentions",
    "n_urls",
    "hashtag_ratio",
    "mention_ratio",

    # emoji / emoticon
    "n_emoji",
    "n_kaomoji",
    "emoji_density",

    # casing style
    "n_all_caps_words",
    "all_caps_ratio",
    "starts_with_uppercase",
    "uppercase_char_ratio",
    "lowercase_char_ratio",

    # informal / slang
    "slang_ratio",
    "has_retweet",              # starts with "RT"
    "n_repeated_chars",         # "loool", "heyyy"

    # punctuation expressiveness
    "n_exclamations",
    "n_questions",
    "consecutive_punct",        # !! ??? !?
    "ellipsis_count",

    # length / density
    "token_count",
    "char_count",
    "avg_token_len",

    # number of distinct scripts (latin vs non-latin chars)
    "non_ascii_ratio",
]


def extract(sentence: str) -> np.ndarray:
    tokens = _TOKEN_RE.findall(sentence)
    n_tokens = max(1, len(tokens))
    n_chars  = max(1, len(sentence))
    lower    = sentence.lower()

    hashtags = _HASHTAG_RE.findall(sentence)
    mentions = _MENTION_RE.findall(sentence)
    urls     = _URL_RE.findall(sentence)
    emojis   = _EMOJI_RE.findall(sentence)
    kaomoji  = _KAOMOJI_RE.findall(sentence)
    all_caps = _ALL_CAPS_RE.findall(sentence)

    slang_words = sum(1 for t in tokens if t.lower() in _SLANG)
    repeated    = _REPEATED_RE.findall(sentence)

    excl = sentence.count("!")
    ques = sentence.count("?")
    consec_punct = len(re.findall(r"[!?]{2,}", sentence))
    ellipsis = sentence.count("...") + sentence.count("…")

    n_non_ascii = sum(1 for c in sentence if ord(c) > 127)

    avg_tok_len = (
        sum(len(t) for t in tokens) / n_tokens if tokens else 0.0
    )

    return np.array([
        float(len(hashtags)),
        float(len(mentions)),
        float(len(urls)),
        len(hashtags) / n_tokens,
        len(mentions) / n_tokens,

        float(len(emojis)),
        float(len(kaomoji)),
        (len(emojis) + len(kaomoji)) / n_tokens,

        float(len(all_caps)),
        len(all_caps) / n_tokens,
        float(sentence[:1].isupper()),
        sum(c.isupper() for c in sentence) / n_chars,
        sum(c.islower() for c in sentence) / n_chars,

        slang_words / n_tokens,
        float(lower.startswith("rt ")),
        float(len(repeated)),

        float(excl),
        float(ques),
        float(consec_punct),
        float(ellipsis),

        float(n_tokens),
        float(n_chars),
        avg_tok_len,

        n_non_ascii / n_chars,
    ], dtype=np.float32)


def extract_batch(sentences) -> np.ndarray:
    n = len(sentences)
    out = np.zeros((n, len(NAMES)), dtype=np.float32)
    for i, s in enumerate(sentences):
        out[i] = extract(s)
    return out
