"""
Classifiers for author switch detection.

Each factory function returns a sklearn Pipeline with StandardScaler + model.
Wrapping in a Pipeline means we never accidentally scale test data with
train statistics — fit() on the pipeline handles everything.

MODEL REGISTRY makes it easy to iterate over all models for comparison.

Notes on choices:
  - LogisticRegression: fast baseline, coefficients are directly interpretable
  - SVM (RBF): strong non-linear baseline, good on small-medium datasets
  - MLP: primary model; kept small (2 hidden layers) to avoid overfitting
    on limited training data and to maintain interpretability via feature
    importance analysis
  - KNN: kept as a sanity-check baseline (no hyperparameters to tune)

All models use random_state=42 for reproducibility.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_logistic_regression(C=1.0):
    """
    Logistic Regression baseline.
    Coefficients give direct feature-level interpretability.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])


def make_svm(C=1.0):
    """
    SVM with RBF kernel.
    probability=True enables predict_proba for threshold tuning.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=C,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])


def make_mlp():
    """
    Small MLP — the primary model.

    Architecture: 2 hidden layers (128, 64 units) with ReLU.
    Kept intentionally small to:
      - avoid overfitting on limited training data,
      - train quickly without a GPU,
      - remain compatible with permutation importance analysis.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),   # 2 layers — per project spec
            activation="relu",
            solver="adam",
            alpha=1e-3,                     # L2 regularisation
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            batch_size=64,
            random_state=42,
            verbose=False,
        )),
    ])


def make_naive_bayes():
    """
    Naive Bayes baseline.
    Very fast; useful as a lower bound / sanity check.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ])


def make_knn(k=5):
    """
    K-Nearest Neighbours baseline.
    No explicit training — useful for debugging feature spaces.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k, metric="euclidean")),
    ])


# Registry: name → factory function.
# Used by evaluation scripts to iterate over all models automatically.
MODEL_REGISTRY = {
    "logistic_regression": make_logistic_regression,
    "svm":                 make_svm,
    "mlp":                 make_mlp,
    "naive_bayes":         make_naive_bayes,
    "knn":                 make_knn,
}


def build_model(name, **kwargs):
    """Instantiate and return the named model (unfitted)."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def train_model(name, X_train, y_train, **kwargs):
    """Instantiate, fit, and return the named model."""
    model = build_model(name, **kwargs)
    model.fit(X_train, y_train)
    print(f"[classifiers] Trained '{name}' on {len(y_train)} samples.")
    return model
