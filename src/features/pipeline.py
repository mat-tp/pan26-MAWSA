"""
Feature pipeline: assembles all feature groups into one sentence vector.
Optimized with true batch extraction, memory efficiency, and caching support.

Design:
  - Primary API is extract_batch(sentences) for vectorized extraction
  - extract_sentence(sentence) still available for single predictions
  - Sentence feature caching with configurable cache directory
  - GPU-ready feature matrices (float32, contiguous memory)

Usage:
    from features.pipeline import FeaturePipeline

    fp = FeaturePipeline()                     # all groups enabled
    fp.fit(all_sentences)                      # fit n-gram models
    
    # Batch extraction (primary API)
    matrix = fp.extract_batch(sentences)       # shape: (n_sents, n_features)
    
    # Single sentence (convenience)
    vec = fp.extract_sentence("This is a sentence.")
"""

import hashlib
import json
import os
import pickle
import numpy as np
from pathlib import Path

from features import (
    char_ngrams,
    function_words,
    lexical,
    pos_features,
    punctuation,
)
from features.ngram_features import NGramExtractor
from utils.config import CACHE_DIR, CACHE_ENABLED

# Available feature groups in extraction order.
# Order matters for ablation reproducibility — do not change without note.
ALL_GROUPS = ["lexical", "punctuation", "function_words", "char_ngrams", "pos", "ngram"]

# Batch size for processing large document collections
DEFAULT_BATCH_SIZE = 500


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
            groups = ALL_GROUPS
        self.groups = groups
        self.use_per_word_fw = use_per_word_fw

        # ---- N‑gram setup ----
        self.ngram_enabled = "ngram" in self.groups and ngram_max_n is not None
        self._ngram_extractor = None
        if self.ngram_enabled:
            self._ngram_extractor = NGramExtractor(
                max_n=ngram_max_n,
                alpha=ngram_alpha,
                tokenizer=tokenizer,
            )
        
        # ---- Caching setup ----
        self.enable_sentence_cache = enable_sentence_cache and CACHE_ENABLED
        self.sentence_cache_dir = sentence_cache_dir or os.path.join(CACHE_DIR, "sentence_features")
        
        # ---- Feature metadata ----
        self.feature_names = self._build_feature_names()
        self.n_features = len(self.feature_names)
        self._fitted = False
        
        print(f"[pipeline] Initialized with {len(self.groups)} groups: {self.groups}")
        print(f"[pipeline] Total features: {self.n_features}")
        if self.enable_sentence_cache:
            print(f"[pipeline] Sentence feature cache: {self.sentence_cache_dir}")

    def _build_feature_names(self):
        """Build descriptive names for all features."""
        names = []
        
        if "lexical" in self.groups:
            names += lexical.NAMES
            
        if "punctuation" in self.groups:
            names += punctuation.NAMES
            
        if "function_words" in self.groups:
            names += function_words.NAMES
            if self.use_per_word_fw:
                names += function_words.get_names_per_word()
                
        if "char_ngrams" in self.groups:
            names += char_ngrams.NAMES
            
        if "pos" in self.groups:
            names += pos_features.NAMES
            
        if self.ngram_enabled and self._ngram_extractor:
            names += self._ngram_extractor.names
            
        return names

    def fit(self, sentences):
        """
        Fit any stateful extractors (currently only n‑gram).
        
        Args:
            sentences: list of strings (all sentences from training data)
        
        Returns:
            self for chaining
        """
        if self.ngram_enabled and self._ngram_extractor:
            print(f"[pipeline] Fitting n-gram models on {len(sentences)} sentences...")
            self._ngram_extractor.fit(sentences)
            print("[pipeline] N-gram models fitted")
        
        self._fitted = True
        return self

    def extract_sentence(self, sentence):
        """
        Extract features for a single sentence (convenience method).
        For bulk processing, use extract_batch() which is much faster.
        
        Args:
            sentence: string sentence to analyze
            
        Returns:
            numpy array of feature values (float32)
        """
        return self.extract_batch([sentence])[0]


    def extract_split(self, sentences):
        """
        Return (dense_features, sparse_features) separately.
        Dense = lexical/punctuation/function_words/pos/ngram
        Sparse = char_ngrams CSR matrix
        """
        from scipy.sparse import csr_matrix
        dense_parts = []
        sparse_part = None

        if "lexical" in self.groups:
            dense_parts.append(lexical.extract_batch(sentences))
        if "punctuation" in self.groups:
            dense_parts.append(punctuation.extract_batch(sentences))
        if "function_words" in self.groups:
            dense_parts.append(
                function_words.extract_batch(
                    sentences,
                    per_word=self.use_per_word_fw
                )
            )
        if "pos" in self.groups:
            dense_parts.append(pos_features.extract_batch(sentences))
        if self.ngram_enabled and self._ngram_extractor:
            dense_parts.append(self._ngram_extractor.extract_batch(sentences))

        if "char_ngrams" in self.groups:
            sparse_part = char_ngrams.extract_batch(sentences)
        else:
            sparse_part = csr_matrix((len(sentences), 0), dtype=np.float32)

        dense = (
            np.ascontiguousarray(
                np.hstack(dense_parts),
                dtype=np.float32
            )
            if dense_parts
            else np.empty((len(sentences), 0), dtype=np.float32)
        )
        return dense, sparse_part


    def extract_batch(self, sentences, batch_size=None, show_progress=False):
        """
        Extract features for a list of sentences with true batch processing.
        
        This is the primary API for feature extraction. Each feature group
        processes all sentences at once for maximum vectorization.
        
        Args:
            sentences: list of strings
            batch_size: sentences per processing chunk (None = auto)
            show_progress: print progress indicators
            
        Returns:
            numpy array of shape (n_sentences, n_features), dtype=float32
        """
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.n_features)
        
        n_sentences = len(sentences)
        
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        
        # For small batches, process directly
        if n_sentences <= batch_size:
            return self._extract_batch_single(sentences, show_progress)
        
        # For large collections, process in chunks
        all_features = []
        n_batches = (n_sentences + batch_size - 1) // batch_size
        
        for i in range(0, n_sentences, batch_size):
            batch = sentences[i:i + batch_size]
            batch_features = self._extract_batch_single(batch, show_progress=False)
            all_features.append(batch_features)
            
            if show_progress and (i // batch_size + 1) % 10 == 0:
                print(f"[pipeline] Processed {min(i + batch_size, n_sentences)}/{n_sentences} sentences")
        
        return np.vstack(all_features)

    def _extract_batch_single(self, sentences, show_progress=False):
        """
        Core batch extraction: process all sentences at once per feature group.

        Routes through extract_split() so that char-ngrams remain in CSR
        format until the very last step.  The final toarray() only touches
        the 12 288-column sparse block once, immediately before hstacking —
        avoiding a transient full-width dense intermediate.

        Args:
            sentences: list of strings
            show_progress: print timing info

        Returns:
            numpy array of shape (n_sentences, n_features), dtype float32
        """
        if show_progress:
            print(
                f"[pipeline] Extracting features for {len(sentences)} sentences..."
            )

        dense, sparse = self.extract_split(sentences)

        # sparse may be a (n, 0) placeholder when char_ngrams is disabled;
        # toarray() on it is free (no allocation).
        if sparse.shape[1] == 0:
            features = dense
        else:
            features = np.hstack([dense, sparse.toarray()])

        return np.ascontiguousarray(features, dtype=np.float32)

    def extract_document(self, sentences, use_cache=True, document_id=None):
        """
        Extract features for a document with caching support.
        
        Args:
            sentences: list of strings
            use_cache: whether to check/use sentence feature cache
            document_id: unique identifier for caching (auto-generated if None)
            
        Returns:
            numpy array of shape (n_sentences, n_features)
        """
        if not sentences:
            return np.array([], dtype=np.float32).reshape(0, self.n_features)
        
        # Try cache if enabled
        if use_cache and self.enable_sentence_cache and document_id:
            cache_path = self._get_sentence_cache_path(document_id)
            if os.path.exists(cache_path):
                if len(sentences) <= 20:  # Only log for small docs to avoid spam
                    print(f"[pipeline] Loading cached features for document {document_id[:16]}...")
                return np.load(cache_path)
        
        # Extract features
        features = self.extract_batch(sentences)
        
        # Save to cache
        if use_cache and self.enable_sentence_cache and document_id:
            cache_path = self._get_sentence_cache_path(document_id)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, features)
            
            if len(sentences) <= 20:
                print(f"[pipeline] Cached features for document {document_id[:16]} at {cache_path}")
        
        return features

    def _get_sentence_cache_path(self, document_id):
        """Get cache file path for a document's sentence features."""
        # Sanitize document_id for filename
        safe_id = hashlib.md5(str(document_id).encode()).hexdigest()[:16]
        return os.path.join(self.sentence_cache_dir, f"{safe_id}.npy")

    def get_feature_names(self):
        """Return list of feature names for interpretability."""
        return self.feature_names

    def describe(self):
        """Print detailed feature group configuration."""
        print("=" * 60)
        print("Feature Pipeline Configuration")
        print("=" * 60)
        print(f"Active groups: {self.groups}")
        print(f"Total features: {self.n_features}")
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
            print(f"  [ngram] {len(self._ngram_extractor.names)} features (max_n={self._ngram_extractor.max_n})")
        print("=" * 60)