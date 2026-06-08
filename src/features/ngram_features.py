"""
N-gram language model features for style change detection.
Extracts perplexity, entropy, and log probability features.
"""

import numpy as np
from collections import defaultdict
import math


class NGramLM:
    """N-gram language model with perplexity and entropy features."""
    
    def __init__(self, n=3):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.total_ngrams = 0
        self.vocab = set()
        self._fitted = False
    
    def fit(self, sentences):
        """Fit n-gram model on training sentences."""
        print(f"[NGram] Fitting {self.n}-gram model on {len(sentences)} sentences...")
        
        for sentence in sentences:
            words = ['<s>'] + sentence.split() + ['</s>']
            
            # Build n-gram counts
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i+self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.total_ngrams += 1
                
                # Add to vocabulary
                for word in ngram:
                    if word not in ['<s>', '</s>']:
                        self.vocab.add(word)
        
        self._fitted = True
        print(f"[NGram] Vocabulary size: {len(self.vocab)}, Total n-grams: {self.total_ngrams}")
        return self
    
    def get_probability(self, word, context):
        """Get smoothed probability P(word | context)."""
        if not self._fitted:
            raise ValueError("Model must be fitted before calling get_probability")
        
        context = tuple(context)
        ngram = context + (word,)
        
        # Laplace smoothing
        numerator = self.ngram_counts.get(ngram, 0) + 1
        denominator = self.context_counts.get(context, 0) + len(self.vocab)
        
        return numerator / denominator
    
    def sentence_log_probability(self, sentence):
        """Calculate log probability of a sentence."""
        if not self._fitted:
            raise ValueError("Model must be fitted before calling sentence_log_probability")
        
        words = ['<s>'] + sentence.split() + ['</s>']
        log_prob = 0.0
        
        for i in range(len(words) - self.n + 1):
            ngram = tuple(words[i:i+self.n])
            context = ngram[:-1]
            word = ngram[-1]
            
            prob = self.get_probability(word, context)
            log_prob += math.log(prob)
        
        return log_prob
    
    def extract_features(self, sentence):
        """Extract n-gram features for a sentence."""
        if not self._fitted:
            raise ValueError("NGramLM must be fitted before extract_features")
        
        log_prob = self.sentence_log_probability(sentence)
        
        # Calculate perplexity: 2^(-log_prob / n_words)
        words = sentence.split()
        n_words = len(words)
        if n_words > 0:
            perplexity = math.exp(-log_prob / n_words) if n_words > 0 else float('inf')
            entropy = -log_prob / n_words if n_words > 0 else float('inf')
        else:
            perplexity = float('inf')
            entropy = float('inf')
        
        return np.array([log_prob, perplexity, entropy], dtype=np.float32)
    
    def __getstate__(self):
        """Custom pickle state to ensure all data is serializable."""
        return {
            'n': self.n,
            'ngram_counts': dict(self.ngram_counts),
            'context_counts': dict(self.context_counts),
            'total_ngrams': self.total_ngrams,
            'vocab': self.vocab,
            '_fitted': self._fitted
        }
    
    def __setstate__(self, state):
        """Custom unpickle state."""
        self.n = state['n']
        self.ngram_counts = defaultdict(int, state['ngram_counts'])
        self.context_counts = defaultdict(int, state['context_counts'])
        self.total_ngrams = state['total_ngrams']
        self.vocab = state['vocab']
        self._fitted = state['_fitted']


class NGramExtractor:
    """Extractor for n-gram LM features (n=1,2,3)."""
    
    def __init__(self, max_n=3):
        self.max_n = max_n
        self.models = {}  # Will store NGramLM for each n
        self._fitted = False
    
    @property
    def names(self):
        """Return feature names for this extractor."""
        names = []
        for n in range(1, self.max_n + 1):
            for metric in ['log_prob', 'perplexity', 'entropy']:
                names.append(f'ngram_{n}_{metric}')
        return names
    
    def fit(self, sentences):
        """Fit n-gram models for n=1 to max_n."""
        print(f"[NGram] Fitting {self.max_n}-gram extractor...")
        
        for n in range(1, self.max_n + 1):
            self.models[n] = NGramLM(n=n)
            self.models[n].fit(sentences)
        
        self._fitted = True
        print(f"[NGram] Extractor fitted with {self.max_n} models")
        return self
    
    def extract_batch(self, sentences):
        """Extract n-gram features for a batch of sentences."""
        if not self._fitted:
            raise ValueError("NGramExtractor must be fitted before extract_batch")
        
        # 3 features per model (log_prob, perplexity, entropy) × max_n
        n_features = 3 * self.max_n
        features = np.zeros((len(sentences), n_features), dtype=np.float32)
        
        for i, sentence in enumerate(sentences):
            for n, model in self.models.items():
                feat = model.extract_features(sentence)
                offset = (n - 1) * 3
                features[i, offset:offset+3] = feat
        
        return features
    
    def __getstate__(self):
        """Custom pickle state."""
        return {
            'max_n': self.max_n,
            'models': self.models,
            '_fitted': self._fitted
        }
    
    def __setstate__(self, state):
        """Custom unpickle state."""
        self.max_n = state['max_n']
        self.models = state['models']
        self._fitted = state['_fitted']

    