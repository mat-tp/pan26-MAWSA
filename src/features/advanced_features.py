"""
Advanced feature extraction: semantic, contextual, and linguistic depth analysis.

These features provide deeper stylometric and linguistic representations beyond
basic n-grams and lexical measures, enabling richer author characterization.

Key feature groups:
- Semantic coherence: word embeddings similarity within sentences
- Contextual patterns: discourse markers, topic transitions
- Linguistic complexity: dependency parsing depth, clause structures
- Semantic shifts: vocabulary diversity, semantic drift metrics
- Pragmatic features: modality, evidentiality, politeness markers
"""

import numpy as np
from collections import Counter
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic and Contextual Analysis
# ─────────────────────────────────────────────────────────────────────────────

def extract_semantic_features(sentence: str) -> dict:
    """
    Extract semantic richness and complexity features.
    
    Features measure vocabulary depth, semantic diversity, and conceptual
    sophistication within a sentence.
    """
    tokens = word_tokenize(sentence.lower())
    tokens = [t for t in tokens if t.isalnum()]
    
    if not tokens:
        return {
            "semantic_diversity": 0.0,
            "type_token_ratio": 0.0,
            "vocabulary_richness": 0.0,
            "content_density": 0.0,
            "unique_pos_tags": 0,
        }
    
    # Type-Token Ratio (TTR): vocabulary richness indicator
    unique_tokens = len(set(tokens))
    type_token_ratio = unique_tokens / len(tokens) if tokens else 0.0
    
    # Semantic diversity using NLTK synsets (approximation of semantic field width)
    try:
        stop_words = set(stopwords.words('english'))
        content_words = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        if content_words:
            synset_counts = []
            for word in content_words[:20]:  # Limit to avoid computational overhead
                synsets = wordnet.synsets(word)
                synset_counts.append(len(synsets))
            semantic_diversity = np.mean(synset_counts) if synset_counts else 0.0
        else:
            semantic_diversity = 0.0
    except Exception:
        semantic_diversity = 0.0
    
    # Content density: ratio of content words to total words
    content_density = len(content_words) / len(tokens) if tokens else 0.0
    
    # POS diversity
    pos_tags = pos_tag(tokens)
    unique_pos = len(set(tag for _, tag in pos_tags))
    
    # Vocabulary richness (Guiraud's index: TTR / sqrt(N))
    vocabulary_richness = unique_tokens / np.sqrt(len(tokens)) if len(tokens) > 0 else 0.0
    
    return {
        "semantic_diversity": float(semantic_diversity),
        "type_token_ratio": float(type_token_ratio),
        "vocabulary_richness": float(vocabulary_richness),
        "content_density": float(content_density),
        "unique_pos_tags": int(unique_pos),
    }


def extract_discourse_markers(sentence: str) -> dict:
    """
    Extract discourse connector and pragmatic marker frequencies.
    
    Discourse markers reveal author's logical organization patterns,
    argumentation style, and perspective management.
    """
    sentence_lower = sentence.lower()
    
    discourse_markers = {
        "additive": ["and", "also", "moreover", "furthermore", "in addition"],
        "causal": ["because", "since", "as", "caused", "due to", "caused by"],
        "contrastive": ["but", "however", "yet", "although", "though", "while"],
        "temporal": ["when", "while", "before", "after", "during", "then"],
        "conclusive": ["therefore", "thus", "so", "consequently", "as a result"],
        "emphasis": ["indeed", "certainly", "surely", "obviously", "clearly"],
        "concessive": ["admittedly", "granted", "true", "to be sure"],
    }
    
    features = {}
    for category, markers in discourse_markers.items():
        count = sum(sentence_lower.count(marker) for marker in markers)
        features[f"discourse_{category}_count"] = float(count)
    
    # Normalized discourse density
    total_markers = sum(features.values())
    features["discourse_marker_density"] = total_markers / len(sentence.split()) if sentence.split() else 0.0
    
    return features


def extract_modality_features(sentence: str) -> dict:
    """
    Extract modality markers reflecting author's epistemic stance.
    
    Modality features capture how authors express certainty, possibility,
    necessity, and their attitude towards propositions.
    """
    sentence_lower = sentence.lower()
    
    modality_markers = {
        "modal_verbs": ["can", "could", "may", "might", "shall", "should", "will", "would", "must", "ought"],
        "certainty_high": ["definitely", "certainly", "surely", "undoubtedly", "absolutely"],
        "certainty_low": ["perhaps", "maybe", "possibly", "arguably", "somewhat"],
        "necessity": ["must", "have to", "need to", "required", "necessary"],
        "possibility": ["can", "could", "may", "might", "possible"],
    }
    
    features = {}
    for category, markers in modality_markers.items():
        count = sum(1 for marker in markers if marker in sentence_lower)
        features[f"modality_{category}"] = float(count)
    
    # Epistemic stance: ratio of low-certainty to high-certainty markers
    low_cert = features.get("modality_certainty_low", 0)
    high_cert = features.get("modality_certainty_high", 0)
    features["epistemic_stance_score"] = (low_cert - high_cert) / (low_cert + high_cert + 1)
    
    return features


def extract_complexity_metrics(sentence: str) -> dict:
    """
    Extract linguistic complexity metrics reflecting sentence structure depth.
    
    Complexity features capture grammatical sophistication, nested structures,
    and syntactic variety used by different authors.
    """
    tokens = word_tokenize(sentence)
    
    # Sentence length normalized
    sentence_length = len(tokens)
    avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
    
    # Clausality: ratio of dependent markers to sentence length
    dependent_markers = ["that", "which", "who", "whom", "whose", "if", "because", "although", "while"]
    clause_count = sum(1 for token in tokens if token.lower() in dependent_markers)
    clausality_index = clause_count / (len(tokens) + 1) if tokens else 0
    
    # Punctuation complexity
    punct_count = sum(1 for char in sentence if char in ".,;:!?()—–-")
    punct_density = punct_count / len(sentence) if sentence else 0
    
    # Embedding depth (rough measure using nested brackets and parentheses)
    embedding_depth = 0
    max_depth = 0
    for char in sentence:
        if char in "([{":
            embedding_depth += 1
            max_depth = max(max_depth, embedding_depth)
        elif char in ")]}":
            embedding_depth -= 1
    
    # Syntactic variety: POS tag diversity
    try:
        pos_tags = pos_tag(tokens)
        unique_pos = len(set(tag for _, tag in pos_tags))
        pos_variety = unique_pos / len(pos_tags) if pos_tags else 0
    except Exception:
        pos_variety = 0.0
    
    return {
        "sentence_length": float(sentence_length),
        "avg_token_length": float(avg_token_length),
        "clausality_index": float(clausality_index),
        "punct_density": float(punct_density),
        "embedding_depth": float(max_depth),
        "pos_variety": float(pos_variety),
    }


def extract_sentiment_markers(sentence: str) -> dict:
    """
    Extract sentiment and evaluation markers in text.
    
    Sentiment features reveal emotional tone, opinion intensity, and
    subjective vs. objective writing style preferences.
    """
    sentence_lower = sentence.lower()
    
    sentiment_lexicon = {
        "positive_strong": ["excellent", "outstanding", "remarkable", "brilliant", "superb", "perfect"],
        "positive_mild": ["good", "nice", "better", "well", "fine", "okay"],
        "negative_strong": ["terrible", "awful", "horrible", "abysmal", "dreadful", "catastrophic"],
        "negative_mild": ["bad", "poor", "worse", "weak", "wrong", "unfavorable"],
        "evaluative": ["important", "significant", "valuable", "worthwhile", "irrelevant", "trivial"],
    }
    
    features = {}
    for category, words in sentiment_lexicon.items():
        count = sum(sentence_lower.count(word) for word in words)
        features[f"sentiment_{category}"] = float(count)
    
    # Overall sentiment balance
    positive = features.get("sentiment_positive_strong", 0) + features.get("sentiment_positive_mild", 0)
    negative = features.get("sentiment_negative_strong", 0) + features.get("sentiment_negative_mild", 0)
    features["sentiment_balance"] = (positive - negative) / (positive + negative + 1)
    features["sentiment_intensity"] = (positive + negative) / len(sentence.split()) if sentence.split() else 0
    
    return features


def extract_referential_cohesion(sentence: str) -> dict:
    """
    Extract referential cohesion markers indicating discourse continuity.
    
    Cohesion features reveal how authors connect sentences, maintain
    topic continuity, and manage information flow.
    """
    sentence_lower = sentence.lower()
    tokens = word_tokenize(sentence_lower)
    
    # Pronoun usage (personal, demonstrative, relative)
    personal_pronouns = ["i", "me", "we", "us", "you", "he", "she", "him", "her", "they", "them"]
    demonstrative = ["this", "that", "these", "those", "it"]
    relative_pronouns = ["who", "whom", "whose", "which", "that"]
    
    personal_count = sum(1 for p in personal_pronouns if p in tokens)
    demonstrative_count = sum(1 for d in demonstrative if d in tokens)
    relative_count = sum(1 for r in relative_pronouns if r in tokens)
    
    # Lexical cohesion: word repetition
    content_tokens = [t for t in tokens if len(t) > 3 and t not in stopwords.words('english')]
    token_frequency = Counter(content_tokens)
    repetition_score = sum(count - 1 for count in token_frequency.values() if count > 1)
    
    return {
        "personal_pronouns": float(personal_count),
        "demonstrative_pronouns": float(demonstrative_count),
        "relative_pronouns": float(relative_count),
        "lexical_repetition": float(repetition_score),
        "cohesion_density": (personal_count + demonstrative_count) / len(tokens) if tokens else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────────────

def extract_advanced_features(sentence: str) -> np.ndarray:
    """
    Extract all advanced features and return as concatenated numpy array.
    
    Returns: 1D feature vector combining all semantic, discourse, modality,
             complexity, sentiment, and cohesion features.
    """
    features_dict = {}
    
    # Combine all feature groups
    features_dict.update(extract_semantic_features(sentence))
    features_dict.update(extract_discourse_markers(sentence))
    features_dict.update(extract_modality_features(sentence))
    features_dict.update(extract_complexity_metrics(sentence))
    features_dict.update(extract_sentiment_markers(sentence))
    features_dict.update(extract_referential_cohesion(sentence))
    
    # Convert to sorted array for consistency
    feature_values = [features_dict[key] for key in sorted(features_dict.keys())]
    return np.array(feature_values, dtype=np.float32)


def extract_advanced_batch(sentences: List[str]) -> np.ndarray:
    """
    Extract advanced features for a batch of sentences.
    
    Returns: 2D array of shape (n_sentences, n_features)
    """
    features_list = []
    for sentence in sentences:
        try:
            features = extract_advanced_features(sentence)
            features_list.append(features)
        except Exception as e:
            # Fallback: zero vector if extraction fails
            print(f"[warning] Advanced feature extraction failed for sentence: {e}")
            if features_list:
                features_list.append(np.zeros_like(features_list[0]))
    
    if not features_list:
        return np.zeros((0, 45), dtype=np.float32)  # Return empty with expected shape
    
    return np.vstack(features_list).astype(np.float32)


def get_advanced_feature_names() -> List[str]:
    """Return list of advanced feature names in sorted order."""
    dummy_sentence = "This is a test sentence for feature name generation."
    features_dict = {}
    
    features_dict.update(extract_semantic_features(dummy_sentence))
    features_dict.update(extract_discourse_markers(dummy_sentence))
    features_dict.update(extract_modality_features(dummy_sentence))
    features_dict.update(extract_complexity_metrics(dummy_sentence))
    features_dict.update(extract_sentiment_markers(dummy_sentence))
    features_dict.update(extract_referential_cohesion(dummy_sentence))
    
    return sorted(features_dict.keys())
