# Author Switch Detection

### PAN 2026 MAWSA — Stylometric Sentence-Level Author Change Detection

A lightweight, interpretable system for detecting author switches between
consecutive sentences in multi-author documents. Built for PAN/CLEF TIRA
evaluation with a focus on handcrafted stylometric features and classical ML.

---

## Design

The goals are:

- Understand _why_ a switch was predicted
- Ablation studies, per-difficulty evaluation, permutation importance

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('universal_tagset'); nltk.download('punkt_tab')"

# Train + evaluate + predict
python app/main.py --mode full --data data/raw

# Only train
python app/main.py --mode train --data data/raw

# Run ablation study
python app/main.py --mode ablation --data data/raw

# Predict on new data (requires trained model)
python app/main.py --mode predict --data data/raw/test --model data/outputs/models/mlp.pkl
```

---

## Feature Groups

| Group            | Features                                                       | Why it works                                             |
| ---------------- | -------------------------------------------------------------- | -------------------------------------------------------- |
| `lexical`        | sentence length, TTR, hapax ratio, Simpson D, word length bins | Author vocabulary habits are consistent across topics    |
| `punctuation`    | comma, semicolon, dash, exclamation counts + density           | Punctuation is habitual and largely topic-independent    |
| `function_words` | ~150 function word frequencies                                 | Closed-class words carry style, not meaning              |
| `char_ngrams`    | char 2/3/4-gram hashes (HashingVectorizer)                     | Sub-word patterns capture spelling and affixation habits |
| `pos`            | Universal POS tag distribution + 12 tracked POS bigrams        | Syntactic preferences (noun-heavy vs verb-heavy, etc.)   |

---

## Pairwise Representation

For consecutive sentences (S*i, S*{i+1}):

```
x_i = |f(S_i) - f(S_{i+1})|
```

A large value in dimension j means the two sentences differ strongly on
feature j — evidence of an author switch. This representation is:

- symmetric (sentence order doesn't matter)
- interpretable (each dimension has a clear meaning)
- compact (same size as a single sentence vector)

---

## Ablation Studies

```python
from evaluation.ablation import run_leave_one_out, run_single_group

# How much does each group contribute?
loo = run_leave_one_out(problems, model_name="mlp")

# What can each group achieve alone?
sgl = run_single_group(problems, model_name="mlp")
```

---

## Feature Importance

```python
from evaluation.importance import permutation_importance, logistic_coefficients

# Model-agnostic: works for any fitted model
imp = permutation_importance(model, X_val, y_val, feature_names=fp.feature_names)

# Direct coefficient interpretation (LR only)
coef = logistic_coefficients(model, feature_names=fp.feature_names)
```

---

## TIRA Submission

1. Train the model and save artefacts:

   ```bash
   python app/main.py --mode train --data data/raw
   cp data/outputs/models/mlp.pkl tira/software/model.pkl
   ```

2. Copy `app/` into `tira/software/`:

   ```bash
   cp -r app tira/software/
   ```

3. Build and test the Docker image:
   ```bash
   docker build -t author-switch-detector -f tira/Dockerfile .
   docker run -e inputDataset=/data/test -e outputDir=/out \
              -v /path/to/test:/data/test -v /path/to/out:/out \
              author-switch-detector
   ```

---

## Reproducibility Checklist

- [x] Fixed random seed (`RANDOM_SEED = 42` in `config.py`)
- [x] StratifiedGroupKFold prevents data leakage across folds
- [x] HashingVectorizer for char n-grams (no fitting on test data)
- [x] All hyperparameters in `config.py`
- [x] Docker image with baked-in NLTK models
- [x] All results saved as JSON/CSV for paper tables

---
