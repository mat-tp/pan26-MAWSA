# API Reference - Enhanced Features

## Advanced Features Module

### Location
```python
from src.features.advanced_features import *
```

### Functions

#### `extract_semantic_features(sentence: str) -> dict`
Extract semantic richness and complexity features.

**Returns**:
```python
{
    "semantic_diversity": float,        # Synset-based semantic field width
    "type_token_ratio": float,          # Vocabulary richness (unique/total)
    "vocabulary_richness": float,       # Guiraud's index
    "content_density": float,           # Content words / total words
    "unique_pos_tags": int,             # Number of distinct POS tags
}
```

---

#### `extract_discourse_markers(sentence: str) -> dict`
Extract discourse connector and pragmatic marker frequencies.

**Returns**:
```python
{
    "discourse_additive_count": float,      # "and", "also", "moreover"
    "discourse_causal_count": float,        # "because", "since", "caused"
    "discourse_contrastive_count": float,   # "but", "however", "yet"
    "discourse_temporal_count": float,      # "when", "before", "after"
    "discourse_conclusive_count": float,    # "therefore", "thus", "so"
    "discourse_emphasis_count": float,      # "indeed", "certainly"
    "discourse_concessive_count": float,    # "admittedly", "granted"
    "discourse_marker_density": float,      # Total markers / word count
}
```

---

#### `extract_modality_features(sentence: str) -> dict`
Extract modality markers reflecting epistemic stance.

**Returns**:
```python
{
    "modality_modal_verbs": float,           # Modal verb count
    "modality_certainty_high": float,        # "definitely", "surely"
    "modality_certainty_low": float,         # "perhaps", "maybe"
    "modality_necessity": float,             # "must", "have to", "required"
    "modality_possibility": float,           # "can", "could", "possible"
    "epistemic_stance_score": float,         # (low - high) / (low + high + 1)
}
```

---

#### `extract_complexity_metrics(sentence: str) -> dict`
Extract linguistic complexity metrics.

**Returns**:
```python
{
    "sentence_length": float,           # Number of tokens
    "avg_token_length": float,          # Average characters per token
    "clausality_index": float,          # Dependent clauses / total
    "punct_density": float,             # Punctuation count / sentence length
    "embedding_depth": float,           # Max nesting depth
    "pos_variety": float,               # Unique POS / total POS
}
```

---

#### `extract_sentiment_markers(sentence: str) -> dict`
Extract sentiment and evaluation markers.

**Returns**:
```python
{
    "sentiment_positive_strong": float,      # "excellent", "outstanding"
    "sentiment_positive_mild": float,        # "good", "nice"
    "sentiment_negative_strong": float,      # "terrible", "awful"
    "sentiment_negative_mild": float,        # "bad", "poor"
    "sentiment_evaluative": float,           # "important", "trivial"
    "sentiment_balance": float,              # (pos - neg) / (pos + neg + 1)
    "sentiment_intensity": float,            # (pos + neg) / word count
}
```

---

#### `extract_referential_cohesion(sentence: str) -> dict`
Extract referential cohesion markers.

**Returns**:
```python
{
    "personal_pronouns": float,         # "I", "we", "they"
    "demonstrative_pronouns": float,    # "this", "that", "it"
    "relative_pronouns": float,         # "who", "which", "that"
    "lexical_repetition": float,        # Word reuse count
    "cohesion_density": float,          # (personal + demonstrative) / tokens
}
```

---

#### `extract_advanced_features(sentence: str) -> np.ndarray`
Extract all advanced features as single vector.

**Parameters**:
- `sentence`: Text string to analyze

**Returns**:
- `np.ndarray`: Shape (45,), dtype float32
  - Concatenates all semantic, discourse, modality, complexity, sentiment, cohesion features

---

#### `extract_advanced_batch(sentences: List[str]) -> np.ndarray`
Batch extraction for multiple sentences.

**Parameters**:
- `sentences`: List of text strings

**Returns**:
- `np.ndarray`: Shape (len(sentences), 45), dtype float32

**Example**:
```python
sentences = ["First sentence.", "Second sentence."]
features = extract_advanced_batch(sentences)  # Shape: (2, 45)
```

---

#### `get_advanced_feature_names() -> List[str]`
Get list of advanced feature names.

**Returns**:
- `List[str]`: Sorted feature names

**Example**:
```python
names = get_advanced_feature_names()
# ['avg_token_length', 'clausality_index', 'cohesion_density', ...]
```

---

## PyTorch Models Module

### Location
```python
from src.models.torch_models import PyTorchClassifier, get_device
```

### Classes

#### `PyTorchClassifier`

GPU-accelerated neural network classifier with sklearn interface.

##### Constructor
```python
PyTorchClassifier(
    model_type: str = "mlp",              # "mlp" or "lstm"
    input_dim: Optional[int] = None,      # Auto-detected during fit
    hidden_dims: List[int] = None,        # [512, 256, 128]
    learning_rate: float = 1e-3,          # Adam learning rate
    batch_size: int = 32,                 # Training batch size
    epochs: int = 50,                     # Max epochs
    dropout_rate: float = 0.3,            # Dropout probability
    batch_norm: bool = True,              # Use batch normalization
    early_stopping: bool = True,          # Stop if val loss plateaus
    patience: int = 10,                   # Epochs to wait
    device: Optional[torch.device] = None,# Auto-detect GPU/CPU
    verbose: bool = True,                 # Print training info
)
```

##### Methods

###### `fit(X, y, X_val=None, y_val=None) -> PyTorchClassifier`
Train the neural network.

**Parameters**:
- `X`: Training features (n_samples, n_features) - ndarray
- `y`: Training labels (n_samples,) with values {0, 1} - ndarray
- `X_val`: Optional validation features
- `y_val`: Optional validation labels

**Returns**: Self (for chaining)

**Example**:
```python
model = PyTorchClassifier()
model.fit(X_train, y_train, X_val, y_val)
```

---

###### `predict(X) -> np.ndarray`
Predict class labels.

**Parameters**:
- `X`: Features (n_samples, n_features)

**Returns**: 
- `np.ndarray`: Shape (n_samples,), values in {0, 1}

---

###### `predict_proba(X) -> np.ndarray`
Predict class probabilities.

**Parameters**:
- `X`: Features (n_samples, n_features)

**Returns**:
- `np.ndarray`: Shape (n_samples, 2)
  - Column 0: P(class=0)
  - Column 1: P(class=1)

---

##### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | PyTorch neural network |
| `device` | torch.device | GPU or CPU |
| `n_features_in_` | int | Input feature dimension |
| `training_history` | dict | Train/val loss per epoch |
| `best_model_state` | dict | Weights from best epoch |

---

### Functions

#### `get_device() -> torch.device`
Get CUDA device if available, else CPU.

**Returns**:
- `torch.device`: CUDA or CPU device

**Example**:
```python
device = get_device()  # "cuda:0" or "cpu"
print(f"Using: {device}")
```

---

#### `make_torch_mlp(...) -> PyTorchClassifier`
Factory for PyTorch MLP.

**Parameters**: Same as PyTorchClassifier

**Returns**: PyTorchClassifier instance

---

#### `make_torch_lstm(...) -> PyTorchClassifier`
Factory for PyTorch LSTM.

**Parameters**: Similar to PyTorchClassifier

**Returns**: PyTorchClassifier instance

---

## Classifiers Module

### Location
```python
from src.models.classifiers import *
```

### Functions

#### `build_model(name, **kwargs) -> estimator`
Build unfitted model.

**Parameters**:
- `name`: Model name (see registry)
- `**kwargs`: Model-specific parameters

**Available Models**:
- `"logistic_regression"`
- `"linear_svc"`
- `"naive_bayes"` ← NEW
- `"mlp"`
- `"random_forest"`
- `"extra_trees"`
- `"xgboost"` (if available)
- `"lightgbm"` (if available)
- `"torch_mlp"` ← NEW (if PyTorch available)
- `"torch_lstm"` ← NEW (if PyTorch available)

**Example**:
```python
model = build_model("torch_mlp", learning_rate=1e-3)
```

---

#### `train_model(name, X_train, y_train, ...) -> estimator`
Build and train model.

**Parameters**:
- `name`: Model name
- `X_train`: Training features
- `y_train`: Training labels
- `use_hyperparam_search`: Enable grid/random search
- `search_method`: "grid" or "randomized"
- `search_cv`: CV folds
- `search_n_iter`: Iterations for randomized search
- `search_scoring`: Scoring metric

**Returns**: Fitted estimator

**Example**:
```python
model = train_model(
    "torch_mlp",
    X_train, y_train,
    use_hyperparam_search=True,
    search_method="randomized"
)
```

---

#### `search_model(name, X_train, y_train, ...) -> tuple`
Hyperparameter search.

**Returns**: `(best_estimator, best_params, search_object)`

---

### Variables

#### `MODEL_REGISTRY: dict`
Maps model names to factory functions.

```python
MODEL_REGISTRY = {
    "logistic_regression": make_logistic_regression,
    "naive_bayes": make_naive_bayes,  # NEW
    "torch_mlp": make_torch_mlp_classifier,  # NEW
    ...
}
```

---

#### `MODEL_DESCRIPTIONS: dict`
Human-readable model descriptions.

```python
MODEL_DESCRIPTIONS["torch_mlp"]  
# "PyTorch MLP with GPU support (Available)"
```

---

#### `HYPERPARAMETER_GRIDS: dict`
Search spaces for hyperparameter tuning.

```python
HYPERPARAMETER_GRIDS["torch_mlp"] = {
    "clf__learning_rate": [1e-4, 1e-3],
    "clf__dropout_rate": [0.2, 0.3],
    "clf__batch_size": [16, 32, 64],
}
```

---

## Configuration Module

### Location
```python
from src.utils.config import *
```

### Variables (New)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CUDA_AVAILABLE` | bool | Auto | GPU detected |
| `USE_ADVANCED_FEATURES` | bool | True | Enable advanced features |
| `USE_GPU_FOR_TREES` | bool | Auto | GPU for XGBoost/LightGBM |
| `USE_GPU_FOR_PYTORCH` | bool | Auto | GPU for PyTorch |
| `PYTORCH_BATCH_SIZE` | int | 32 | Training batch size |
| `PYTORCH_EPOCHS` | int | 100 | Max training epochs |
| `PYTORCH_LEARNING_RATE` | float | 1e-3 | Adam LR |
| `PYTORCH_DROPOUT` | float | 0.3 | Dropout rate |
| `PYTORCH_HIDDEN_DIMS` | list | [512, 256, 128] | Hidden layer sizes |

---

## Feature Pipeline Module

### Location
```python
from src.features.pipeline import FeaturePipeline
```

### Classes

#### `FeaturePipeline`

Main feature extraction interface.

##### Methods

###### `fit(sentences: List[str]) -> FeaturePipeline`
Fit n-gram models and other trainable components.

###### `extract_batch(sentences: List[str]) -> np.ndarray`
Extract features for multiple sentences.

**Parameters**:
- `sentences`: List of text strings

**Returns**:
- `np.ndarray`: Shape (len(sentences), n_features)

###### `extract_sentence(sentence: str) -> np.ndarray`
Extract features for single sentence.

**Parameters**:
- `sentence`: Text string

**Returns**:
- `np.ndarray`: Shape (n_features,)

---

### Variables (New)

| Variable | Type | Values |
|----------|------|--------|
| `ALL_GROUPS` | list | [..., "advanced", "embeddings"] |
| `USE_ADVANCED_FEATURES` | bool | See config |

---

## Example Workflows

### Complete Training & Evaluation

```python
from src.features.pipeline import FeaturePipeline
from src.models.classifiers import train_model, compare_all_models
from src.utils.config import MODELS_TO_COMPARE
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load data
X_text = [...]  # List of sentence pairs
y = np.array([...])  # Binary labels

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X_text, y)

# 3. Extract features
fp = FeaturePipeline()
fp.fit(X_train)
X_train_feat = fp.extract_batch(X_train)
X_test_feat = fp.extract_batch(X_test)

# 4. Train all models
results = compare_all_models(X_train_feat, y_train)

# 5. Train best model
best = train_model("torch_mlp", X_train_feat, y_train)

# 6. Evaluate
score = best.score(X_test_feat, y_test)
print(f"Test F1: {score:.4f}")
```

---

## Error Handling

### PyTorch Not Available
```python
try:
    from src.models.classifiers import build_model
    model = build_model("torch_mlp")
except KeyError:
    print("PyTorch models not available. Install PyTorch first.")
```

### GPU Not Available
```python
from src.utils.config import CUDA_AVAILABLE

if not CUDA_AVAILABLE:
    print("GPU not available. Training on CPU (will be slower).")
```

### Feature Extraction Failure
```python
from src.features.advanced_features import extract_advanced_batch

try:
    features = extract_advanced_batch(sentences)
except Exception as e:
    print(f"Feature extraction failed: {e}")
    # Fallback: use base features only
```

---

## Performance Tips

1. **GPU Utilization**: Ensure CUDA_AVAILABLE=True
2. **Large Datasets**: Increase batch_size (32→64) for speed
3. **Memory Constraints**: Decrease hidden_dims or batch_size
4. **Training Speed**: Use fewer epochs and larger learning rate
5. **Accuracy**: Use early_stopping=True and validation split

---

## Summary

- **45 Advanced Features** covering semantic, discourse, modality, complexity, sentiment, cohesion
- **9 Classifiers** including Naïve Bayes and PyTorch neural networks
- **GPU Acceleration** for XGBoost, LightGBM, and PyTorch
- **Sklearn Compatible** for pipelines and cross-validation
- **Production Ready** with serialization and error handling
