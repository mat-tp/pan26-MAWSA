"""
Feature pipeline: assembles all feature groups into one sentence vector.

Design:
  - Each feature group is a module with an extract(sentence) function.
  - This pipeline calls them in order and concatenates the results.
  - Feature groups can be toggled on/off for ablation studies.
  - The pipeline is stateless (no fitting required) except for char n-grams,
    which use HashingVectorizer internally (also stateless).

Usage:
    from features.pipeline import FeaturePipeline

    fp = FeaturePipeline()                     # all groups enabled
    fp = FeaturePipeline(use_pos=False)        # skip POS (faster)
    fp = FeaturePipeline(groups=["lexical", "punctuation"])  # only these groups

    vec = fp.extract_sentence("This is a sentence.")
    matrix = fp.extract_document(sentences)    # shape: (n_sents, n_features)
"""

import numpy as np

from app.features import (
    char_ngrams,
    function_words,
    lexical,
    pos_features,
    punctuation,
)

# Available feature groups in extraction order.
# Order matters for ablation reproducibility — do not change without note.
ALL_GROUPS = ["lexical", "punctuation", "function_words", "char_ngrams", "pos"]


class FeaturePipeline:
    """
    Assembles sentence feature vectors from configurable feature groups.

    Parameters
    ----------
    groups : list of str, optional
        Which feature groups to include.  Defaults to ALL_GROUPS.
        Valid values: "lexical", "punctuation", "function_words",
                      "char_ngrams", "pos"
    use_per_word_fw : bool
        If True, include per-function-word frequencies (adds ~150 features).
        Default False for a more compact representation.
    """

    def __init__(self, groups=None, use_per_word_fw=False):
        if groups is None:
            groups = ALL_GROUPS
        self.groups = groups
        self.use_per_word_fw = use_per_word_fw

        # Build the feature name list once so downstream code can label axes.
        self.feature_names = self._build_feature_names()
        self.n_features = len(self.feature_names)

    def _build_feature_names(self):
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
        return names

    def extract_sentence(self, sentence):
        """Return a 1-D float32 numpy array for one sentence."""
        parts = []

        if "lexical" in self.groups:
            parts.extend(lexical.extract(sentence))

        if "punctuation" in self.groups:
            parts.extend(punctuation.extract(sentence))

        if "function_words" in self.groups:
            parts.extend(function_words.extract(sentence))
            if self.use_per_word_fw:
                parts.extend(function_words.extract_per_word(sentence))

        if "char_ngrams" in self.groups:
            parts.append(char_ngrams.extract(sentence))

        if "pos" in self.groups:
            parts.extend(pos_features.extract(sentence))

        # Some groups return numpy arrays, others plain lists — flatten all.
        vec = []
        for p in parts:
            if hasattr(p, "__iter__") and not isinstance(p, (str, bytes)):
                vec.extend(float(x) for x in p)
            else:
                vec.append(float(p))

        return np.array(vec, dtype=np.float32)

    def extract_document(self, sentences):
        """
        Extract features for every sentence in a document.

        Returns a 2-D array of shape (n_sentences, n_features).
        """
        return np.stack([self.extract_sentence(s) for s in sentences], axis=0)

    def describe(self):
        """Print a summary of the active feature groups and dimensions."""
        print(
            f"FeaturePipeline — {len(self.groups)} group(s), "
            f"{self.n_features} features total"
        )
        for g in self.groups:
            print(f"  [{g}]")
        if self.use_per_word_fw:
            print("  [function_words: per-word frequencies ON]")
