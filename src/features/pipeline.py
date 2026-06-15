"""
Feature pipeline: assembles all feature groups into one sentence vector.

Primary API:
    fp = FeaturePipeline()           # all groups enabled
    fp.fit(all_sentences)            # fit n-gram models (if ngram group active)
    matrix = fp.extract_batch(sents) # shape (n_sents, n_features), float32
    vec    = fp.extract_sentence(s)  # single-sentence convenience wrapper
"""

import hashlib
import json
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold

from features import (
    char_ngrams,
    embeddings,
    function_words,
    lexical,
    pos_features,
    punctuation,
    readability,
    syntax_features,
    social_media_features,
    advanced_features,
)
from features.ngram_features import NGramExtractor
from utils.config import CACHE_DIR, CACHE_ENABLED, USE_ADVANCED_FEATURES

# Available feature groups in extraction order.
# Order matters for ablation reproducibility — do not change without note.
ALL_GROUPS = [
    "lexical",
    "punctuation",
    "function_words",
    "readability",
    "char_ngrams",
    "pos",
    "ngram",
    "syntax",
    "social_media",
    "advanced",
    "embeddings",
]

# Default active groups if none are supplied. Embeddings and advanced are opt-in.
DEFAULT_GROUPS = [
    "lexical",
    "punctuation",
    "function_words",
    "readability",
    "char_ngrams",
    "pos",
    "ngram",
    "syntax",
    "social_media",
]

# Batch size for processing large document collections
DEFAULT_BATCH_SIZE = 500


class FeatureSelectorWithNames:
    """VarianceThreshold wrapper that tracks which feature names survive."""
    
    def __init__(self, threshold=0.01):
        self.selector = VarianceThreshold(threshold=threshold)
        self.support_ = None
        self.threshold = threshold
    
    def fit(self, X):
        self.selector.fit(X)
        self.support_ = self.selector.get_support()
        return self
    
    def transform(self, X):
        return self.selector.transform(X)
    
    def fit_transform(self, X):
        X2 = self.selector.fit_transform(X)
        self.support_ = self.selector.get_support()
        return X2
    
    def get_support(self):
        return self.support_


class FeaturePipeline:
    """Feature extraction pipeline for sentence-level stylometric features."""
    
    def __init__(self, groups=None, use_per_word_fw=False,
                 ngram_max_n=3, ngram_alpha=1.0, tokenizer=None,
                 sentence_cache_dir=None, enable_sentence_cache=True):
        """
        Initialize feature pipeline.
        
        Args:
            groups: List of feature groups to use (None = all groups)
            use_per_word_fw: Include per-function-word frequencies
            ngram_max_n: Maximum n-gram order for perplexity features
            ngram_alpha: Smoothing parameter for n-gram LM
            tokenizer: Custom tokenizer for n-gram features
            sentence_cache_dir: Directory for sentence feature caching
            enable_sentence_cache: Whether to use sentence feature cache
        """
        if groups is None:
            groups = DEFAULT_GROUPS
        self.groups = groups
        self.use_per_word_fw = use_per_word_fw

        # ---- N‑gram setup ----
        self.ngram_enabled = "ngram" in self.groups and ngram_max_n is not None
        self._ngram_extractor = None
        if self.ngram_enabled:
            self._ngram_extractor = NGramExtractor(
                max_n=ngram_max_n,
            )
        
        # ---- Caching setup ----
        self.enable_sentence_cache = enable_sentence_cache and CACHE_ENABLED
        self.sentence_cache_dir = sentence_cache_dir or os.path.join(CACHE_DIR, "sentence_features")
        
        # ---- Feature metadata ----
        self.n_features = None
        self.feature_names = []
        self._final_feature_names = None  # Set after variance filtering
        
        self._fitted = False
        
        print(f"[pipeline] Initialized with {len(self.groups)} groups: {self.groups}")
        if self.enable_sentence_cache:
            print(f"[pipeline] Sentence feature cache: {self.sentence_cache_dir}")

    @property
    def is_fitted(self):
        """Return whether the pipeline has been fitted."""
        return self._fitted
    
    def set_final_feature_names(self, names):
        """Set the final feature names after variance filtering."""
        self._final_feature_names = list(names)
    
    def get_final_feature_names(self):
        """Return final feature names (after filtering) or original names."""
        if self._final_feature_names is not None:
            return self._final_feature_names
        return self.get_feature_names()
    
    def _lock_dimensions(self):
        """Lock feature dimensions and build proper feature names."""
        probe_dense, probe_sparse = self.extract_split(
            ["bootstrap sentence"],
            validate=False
        )

        n_dense = probe_dense.shape[1]
        n_sparse = probe_sparse.shape[1]
        self.n_features = n_dense + n_sparse

        # Build feature names from each group
        self.feature_names = self._build_feature_names()
        
        # If names don't match, build descriptive fallback names
        if not self.feature_names or len(self.feature_names) != self.n_features:
            print(f"[pipeline] Name count ({len(self.feature_names)}) != feature count ({self.n_features})")
            print(f"[pipeline] Building descriptive fallback names...")
            self.feature_names = self._build_descriptive_fallback_names(n_dense, n_sparse)

        print(
            f"[pipeline] Locked dimensions → "
            f"dense={n_dense} "
            f"sparse={n_sparse} "
            f"total={self.n_features}"
        )
        if len(self.feature_names) >= 3:
            print(f"[pipeline] Feature names sample: {self.feature_names[:3]}...{self.feature_names[-3:]}")

    def _build_feature_names(self):
        """Build feature names from active groups in extraction order."""
        names = []

        # Dense features (in extraction order from extract_split)
        if "lexical" in self.groups:
            names += [f"lexical_{x}" for x in lexical.NAMES]

        if "punctuation" in self.groups:
            names += [f"punctuation_{x}" for x in punctuation.NAMES]

        if "function_words" in self.groups:
            if self.use_per_word_fw:
                names += function_words.get_names_per_word()
            else:
                names += function_words.NAMES

        if "readability" in self.groups:
            names += readability.NAMES

        if "pos" in self.groups:
            names += pos_features.NAMES

        if self.ngram_enabled and self._ngram_extractor and getattr(self._ngram_extractor, '_fitted', False):
            names += getattr(self._ngram_extractor, "names", [])

        if "syntax" in self.groups:
            names += syntax_features.NAMES

        if "social_media" in self.groups:
            names += social_media_features.NAMES

        if "advanced" in self.groups:
            names += advanced_features.get_advanced_feature_names()

        if "embeddings" in self.groups:
            names += embeddings.get_feature_names()

        # Sparse features (char_ngrams are always last in extract_split)
        if "char_ngrams" in self.groups:
            names += char_ngrams.NAMES

        return names

    def _build_descriptive_fallback_names(self, n_dense, n_sparse):
        """Build descriptive fallback names when group names don't match counts."""
        names = []
        
        # Try to get names from groups for dense features
        dense_names = []
        if "lexical" in self.groups:
            dense_names += [f"lexical_{x}" for x in lexical.NAMES]
        if "punctuation" in self.groups:
            dense_names += [f"punctuation_{x}" for x in punctuation.NAMES]
        if "function_words" in self.groups:
            if self.use_per_word_fw:
                dense_names += function_words.get_names_per_word()
            else:
                dense_names += function_words.NAMES
        if "readability" in self.groups:
            dense_names += readability.NAMES
        if "pos" in self.groups:
            dense_names += pos_features.NAMES
        if self.ngram_enabled and self._ngram_extractor and getattr(self._ngram_extractor, '_fitted', False):
            dense_names += getattr(self._ngram_extractor, "names", [])
        if "syntax" in self.groups:
            dense_names += syntax_features.NAMES
        if "social_media" in self.groups:
            dense_names += social_media_features.NAMES
        if "advanced" in self.groups:
            dense_names += advanced_features.get_advanced_feature_names()
        if "embeddings" in self.groups:
            dense_names += embeddings.get_feature_names()
        
        # Use available names, fill remaining with descriptive placeholders
        for i in range(n_dense):
            if i < len(dense_names):
                names.append(dense_names[i])
            else:
                names.append(f"dense_extra_{i - len(dense_names)}")
        
        # Sparse names
        sparse_names = char_ngrams.NAMES if "char_ngrams" in self.groups else []
        for i in range(n_sparse):
            if i < len(sparse_names):
                names.append(sparse_names[i])
            else:
                names.append(f"char_ngram_{i}")
        
        return names

    def fit(self, sentences):
        """Fit all trainable components (n-gram models, embeddings PCA, etc.)."""
        print(f"[Pipeline] Fitting on {len(sentences)} sentences...")

        # Fit n-gram extractor
        if hasattr(self, '_ngram_extractor') and self._ngram_extractor:
            print(f"[Pipeline] Fitting n-gram extractor...")
            self._ngram_extractor.fit(sentences)

        # Fit embeddings PCA transformation BEFORE extraction
        if "embeddings" in self.groups:
            print(f"[Pipeline] EXECUTING embeddings.fit()...")
            try:
                embeddings.fit(sentences)
                print(f"[Pipeline] embeddings.fit() completed successfully")
            except Exception as e:
                print(f"[Pipeline] WARNING: embeddings.fit() failed: {e}")

        # Lock dimensions after fitting
        self._lock_dimensions()
        
        self._fitted = True
        print(f"[Pipeline] Fitted successfully")
        return self

    def extract_sentence(self, sentence):
        """Extract features for a single sentence."""
        return self.extract_batch([sentence])[0]

    def extract_split(self, sentences, validate=True):
        """Extract features returning separate dense and sparse matrices."""
        from scipy.sparse import csr_matrix

        dense_parts = []

        def add(x):
            if x is None:
                return
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            dense_parts.append(x)

        if "lexical" in self.groups:
            add(lexical.extract_batch(sentences))
        if "punctuation" in self.groups:
            add(punctuation.extract_batch(sentences))
        if "function_words" in self.groups:
            add(function_words.extract_batch(sentences, per_word=self.use_per_word_fw))
        if "readability" in self.groups:
            add(readability.extract_batch(sentences))
        if "pos" in self.groups:
            add(pos_features.extract_batch(sentences))
        if self.ngram_enabled and self._ngram_extractor and getattr(self._ngram_extractor, '_fitted', False):
            add(self._ngram_extractor.extract_batch(sentences))
        if "syntax" in self.groups:
            add(syntax_features.extract_batch(sentences))
        if "social_media" in self.groups:
            add(social_media_features.extract_batch(sentences))
        if "advanced" in self.groups and USE_ADVANCED_FEATURES:
            add(advanced_features.extract_advanced_batch(sentences))
        if "embeddings" in self.groups:
            add(embeddings.extract_batch(sentences))

        dense = np.hstack(dense_parts) if dense_parts else np.empty((len(sentences), 0), dtype=np.float32)
        dense = np.ascontiguousarray(dense, dtype=np.float32)

        sparse = (
            char_ngrams.extract_batch(sentences)
            if "char_ngrams" in self.groups
            else csr_matrix((len(sentences), 0), dtype=np.float32)
        )

        if validate and self.n_features is not None:
            actual = dense.shape[1] + sparse.shape[1]
            if actual != self.n_features:
                raise RuntimeError(f"Feature mismatch: expected={self.n_features}, actual={actual}")

        return dense, sparse

    def extract_batch(self, sentences, batch_size=None, show_progress=False):
        """Extract features for a list of sentences."""
        if self.n_features is None:
            self._lock_dimensions()
            
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.n_features)
        
        n_sentences = len(sentences)
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        
        if n_sentences <= batch_size:
            return self._extract_batch_single(sentences, show_progress)
        
        all_features = []
        for i in range(0, n_sentences, batch_size):
            batch = sentences[i:i + batch_size]
            batch_features = self._extract_batch_single(batch, show_progress=False)
            all_features.append(batch_features)
            if show_progress and (i // batch_size + 1) % 10 == 0:
                print(f"[pipeline] Processed {min(i + batch_size, n_sentences)}/{n_sentences} sentences")
        
        return np.vstack(all_features)

    def _extract_batch_single(self, sentences, show_progress=False):
        """Core batch extraction."""
        if show_progress:
            print(f"[pipeline] Extracting features for {len(sentences)} sentences...")
        dense, sparse = self.extract_split(sentences)
        if sparse.shape[1] == 0:
            features = dense
        else:
            features = np.hstack([dense, sparse.toarray()])
        return np.ascontiguousarray(features, dtype=np.float32)

    def extract_document(self, sentences, use_cache=False, document_id=None):
        """Extract features for a document with caching support."""
        if self.n_features is None:
            self._lock_dimensions()
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.n_features)
        if use_cache and self.enable_sentence_cache and document_id:
            cache_path = self._get_sentence_cache_path(document_id)
            if os.path.exists(cache_path):
                return np.load(cache_path)
        features = self.extract_batch(sentences)
        if use_cache and self.enable_sentence_cache and document_id:
            cache_path = self._get_sentence_cache_path(document_id)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, features)
        return features

    def _get_sentence_cache_path(self, document_id):
        safe_id = hashlib.md5(str(document_id).encode()).hexdigest()[:16]
        return os.path.join(self.sentence_cache_dir, f"{safe_id}.npy")

    def get_feature_names(self):
        """Return list of feature names."""
        if not self.feature_names:
            self._lock_dimensions()
        return self.feature_names

    def describe(self):
        """Print detailed feature group configuration."""
        print("=" * 60)
        print("Feature Pipeline Configuration")
        print("=" * 60)
        print(f"Active groups: {self.groups}")
        print(f"Total features: {self.n_features if self.n_features is not None else 'Not locked yet'}")
        print(f"Fitted: {self._fitted}")
        print(f"Sentence cache: {'Enabled' if self.enable_sentence_cache else 'Disabled'}")
        if "lexical" in self.groups:
            print(f"  [lexical] {len(lexical.NAMES)} features (batch)")
        if "punctuation" in self.groups:
            print(f"  [punctuation] {len(punctuation.NAMES)} features (batch)")
        if "function_words" in self.groups:
            fw_count = len(function_words.NAMES)
            if self.use_per_word_fw:
                fw_count += len(function_words.get_names_per_word())
            print(f"  [function_words] {fw_count} features (batch)")
        if "char_ngrams" in self.groups:
            print(f"  [char_ngrams] {len(char_ngrams.NAMES)} features (batch)")
        if "pos" in self.groups:
            print(f"  [pos] {len(pos_features.NAMES)} features (batch)")
        if self.ngram_enabled and self._ngram_extractor:
            ngram_count = len(self._ngram_extractor.names) if self._ngram_extractor._fitted else "not fitted"
            print(f"  [ngram] {ngram_count} features (max_n={self._ngram_extractor.max_n})")
        if "syntax" in self.groups:
            print(f"  [syntax] {len(syntax_features.NAMES)} features (batch)")
        if "social_media" in self.groups:
            print(f"  [social_media] {len(social_media_features.NAMES)} features (batch)")
        print("=" * 60)

    def get_dimension_split(self):
        """Return (n_dense, n_sparse) after dimensions are locked."""
        if self.n_features is None:
            self._lock_dimensions()
        probe_dense, probe_sparse = self.extract_split(["bootstrap sentence"], validate=False)
        return probe_dense.shape[1], probe_sparse.shape[1]

    def save(self, path):
        """Save entire fitted pipeline."""
        import cloudpickle
        with open(path, 'wb') as f:
            cloudpickle.dump(self, f)
        print(f"[Pipeline] Saved to {path}")
    
    @staticmethod
    def load(path):
        """Load fitted pipeline."""
        import cloudpickle
        with open(path, 'rb') as f:
            pipeline = cloudpickle.load(f)
        print(f"[Pipeline] Loaded from {path}")
        return pipeline