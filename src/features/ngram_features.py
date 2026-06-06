"""
N-gram language model features.

Features:
    - Entropy
    - Perplexity
    - Average log probability

Supports:
    unigram → max_n gram

Uses Add-alpha smoothing.
"""

import math
from collections import Counter, defaultdict

import numpy as np


BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


class NGramLM:
    """
    N-gram language model with Add-alpha smoothing.
    """

    def __init__(self, n=3, alpha=1.0):
        self.n = n
        self.alpha = alpha

        self.vocab = set()
        self.vocab_size = 0

        self.ngram_counts = defaultdict(Counter)
        self.context_totals = Counter()

    # -------------------------------------------------
    # TRAINING
    # -------------------------------------------------

    def fit(self, tokenized_sentences):

        token_freq = Counter()

        for sent in tokenized_sentences:
            token_freq.update(sent)

        processed = []

        for sent in tokenized_sentences:

            sent = [
                token
                if token_freq[token] > 1
                else UNK
                for token in sent
            ]

            padded = (
                [BOS] * (self.n - 1)
                + sent
                + [EOS]
            )

            processed.append(padded)

        self.vocab = set()

        for sent in processed:

            self.vocab.update(sent)

            for i in range(len(sent) - self.n + 1):

                context = tuple(
                    sent[i : i + self.n - 1]
                )

                word = sent[i + self.n - 1]

                self.ngram_counts[context][word] += 1
                self.context_totals[context] += 1

        self.vocab.add(UNK)
        self.vocab_size = len(self.vocab)

        return self

    # -------------------------------------------------
    # PROBABILITY
    # -------------------------------------------------

    def prob(self, word, context):

        word = (
            word
            if word in self.vocab
            else UNK
        )

        ctx_counter = self.ngram_counts.get(context)

        count = (
            ctx_counter.get(word, 0)
            if ctx_counter is not None
            else 0
        )

        total = self.context_totals.get(context, 0)

        return (
            count + self.alpha
        ) / (
            total + self.alpha * self.vocab_size
        )

    # -------------------------------------------------
    # EVALUATION
    # -------------------------------------------------

    def sentence_log_prob(self, tokens):

        tokens = [
            t if t in self.vocab else UNK
            for t in tokens
        ]

        tokens = (
            [BOS] * (self.n - 1)
            + tokens
            + [EOS]
        )

        log_prob = 0.0

        for i in range(
            self.n - 1,
            len(tokens)
        ):

            context = tuple(
                tokens[
                    i - self.n + 1 : i
                ]
            )

            word = tokens[i]

            log_prob += math.log(
                self.prob(word, context)
            )

        return log_prob

    def avg_log_prob(self, tokens):

        N = max(
            1,
            len(tokens) + 1
        )

        return (
            self.sentence_log_prob(tokens)
            / N
        )

    def entropy(self, tokens):

        avg_lp = self.avg_log_prob(tokens)

        return -avg_lp / math.log(2)

    def perplexity(self, tokens):

        return 2 ** self.entropy(tokens)


class NGramExtractor:
    """
    Extracts LM-based features.

    Per n:
        entropy
        perplexity
        avg_log_prob
    """

    def __init__(
        self,
        max_n=3,
        alpha=1.0,
        tokenizer=None,
    ):

        self.max_n = max_n
        self.alpha = alpha

        self.tokenizer = (
            tokenizer
            if tokenizer
            else lambda s: s.split()
        )

        self.models = []

    def fit(self, sentences):

        tokenized = [
            self.tokenizer(s)
            for s in sentences
        ]

        self.models = []

        for n in range(
            1,
            self.max_n + 1
        ):

            lm = NGramLM(
                n=n,
                alpha=self.alpha,
            )

            lm.fit(tokenized)

            self.models.append(lm)

        return self

    def extract(self, sentence):

        tokens = self.tokenizer(sentence)

        features = []

        for lm in self.models:

            features.extend(
                [
                    lm.entropy(tokens),
                    lm.perplexity(tokens),
                    lm.avg_log_prob(tokens),
                ]
            )

        return np.asarray(
            features,
            dtype=np.float32,
        )

    @property
    def names(self):

        names = []

        for n in range(
            1,
            self.max_n + 1
        ):

            names.extend(
                [
                    f"ngram_{n}_entropy",
                    f"ngram_{n}_ppl",
                    f"ngram_{n}_avg_logprob",
                ]
            )

        return names
    def extract_batch(self, sentences):
        """
        Vectorised batch extraction.

        Pre-allocates the output array and fills it row-by-row, eliminating
        the per-sentence list-append and final vstack overhead of the old
        np.vstack([self.extract(s) for s in sentences]) pattern.
        """
        n_feats = len(self.names)
        n       = len(sentences)
        out     = np.empty((n, n_feats), dtype=np.float32)
        for i, s in enumerate(sentences):
            out[i] = self.extract(s)
        return out