"""
N-gram language model features.

Extracts entropy, perplexity, and average log-probability for each order
from unigram up to max_n. Uses add-alpha smoothing.
"""

import math
from collections import Counter, defaultdict

import numpy as np

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"


class NGramLM:
    """N-gram language model with add-alpha smoothing."""

    def __init__(self, n=3, alpha=1.0):
        self.n     = n
        self.alpha = alpha
        self.vocab       = set()
        self.vocab_size  = 0
        self.ngram_counts   = defaultdict(Counter)
        self.context_totals = Counter()

    def fit(self, tokenized_sentences):
        """Train on a list of already-tokenised sentences."""
        token_freq = Counter(t for sent in tokenized_sentences for t in sent)

        # Replace singletons with UNK, then pad with BOS/EOS
        processed = []
        for sent in tokenized_sentences:
            sent   = [t if token_freq[t] > 1 else UNK for t in sent]
            padded = [BOS] * (self.n - 1) + sent + [EOS]
            processed.append(padded)

        for sent in processed:
            self.vocab.update(sent)
            for i in range(len(sent) - self.n + 1):
                context = tuple(sent[i: i + self.n - 1])
                word    = sent[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.context_totals[context]     += 1

        self.vocab.add(UNK)
        self.vocab_size = len(self.vocab)
        return self

    def prob(self, word, context):
        """Smoothed probability P(word | context)."""
        word       = word if word in self.vocab else UNK
        ctx        = self.ngram_counts.get(context)
        count      = ctx.get(word, 0) if ctx is not None else 0
        total      = self.context_totals.get(context, 0)
        return (count + self.alpha) / (total + self.alpha * self.vocab_size)

    def sentence_log_prob(self, tokens):
        """Sum of log P(word | context) over all positions."""
        tokens = [t if t in self.vocab else UNK for t in tokens]
        tokens = [BOS] * (self.n - 1) + tokens + [EOS]
        return sum(
            math.log(self.prob(tokens[i], tuple(tokens[i - self.n + 1: i])))
            for i in range(self.n - 1, len(tokens))
        )

    def avg_log_prob(self, tokens):
        """sentence_log_prob normalised by token count."""
        return self.sentence_log_prob(tokens) / max(1, len(tokens) + 1)

    def entropy(self, tokens):
        """Entropy in bits (negated average log2 probability)."""
        return -self.avg_log_prob(tokens) / math.log(2)

    def perplexity(self, tokens):
        """Perplexity = 2^entropy."""
        return 2 ** self.entropy(tokens)


class NGramExtractor:
    """
    Extracts LM-based stylometric features.

    Per order n (1..max_n) produces: entropy, perplexity, avg_log_prob.
    """

    def __init__(self, max_n=3, alpha=1.0, tokenizer=None):
        self.max_n     = max_n
        self.alpha     = alpha
        self.tokenizer = tokenizer if tokenizer else str.split
        self.models    = []

    def fit(self, sentences):
        """Fit one NGramLM per order on a list of sentences."""
        tokenized  = [self.tokenizer(s) for s in sentences]
        self.models = [NGramLM(n=n, alpha=self.alpha).fit(tokenized) for n in range(1, self.max_n + 1)]
        return self

    def extract(self, sentence):
        """Extract LM features for a single sentence."""
        tokens   = self.tokenizer(sentence)
        features = []
        for lm in self.models:
            features.extend([lm.entropy(tokens), lm.perplexity(tokens), lm.avg_log_prob(tokens)])
        return np.asarray(features, dtype=np.float32)

    @property
    def names(self):
        """Feature name list aligned with extract() output."""
        names = []
        for n in range(1, self.max_n + 1):
            names.extend([f"ngram_{n}_entropy", f"ngram_{n}_ppl", f"ngram_{n}_avg_logprob"])
        return names

    def extract_batch(self, sentences):
        """Vectorised batch extraction over a list of sentences."""
        n_feats = len(self.names)
        out     = np.empty((len(sentences), n_feats), dtype=np.float32)
        for i, s in enumerate(sentences):
            out[i] = self.extract(s)
        return out