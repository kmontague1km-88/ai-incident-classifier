"""AI layer: TF-IDF features + multinomial logistic regression.

Word n-grams capture phrases ("replica lag", "failed login"); character
n-grams make the model tolerant of host names, typos, and jargon it has not
seen. Logistic regression gives calibrated-enough probabilities for a
confidence threshold and per-word weights that explain each decision.
"""
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from . import config as C


def build_pipeline():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def train(texts, labels):
    pipe = build_pipeline()
    pipe.fit(texts, labels)
    return pipe


def save(pipe, path=C.MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def load(path=C.MODEL_PATH):
    # Model file is built by our own CI pipeline and shipped inside the
    # deployment image; it is never loaded from user input.
    return joblib.load(path)


@dataclass
class Prediction:
    category: str
    confidence: float
    runner_up: str
    top_terms: list


def predict(pipe, text, n_terms=3):
    proba = pipe.predict_proba([text])[0]
    classes = pipe.classes_
    order = np.argsort(proba)[::-1]
    best, second = order[0], order[1]
    return Prediction(classes[best], float(proba[best]), classes[second], explain(pipe, text, classes[best], n_terms))


def explain(pipe, text, category, n_terms=3):
    """Word features in this alert that pushed hardest toward the category."""
    words = pipe.named_steps["features"].transformer_list[0][1]
    clf = pipe.named_steps["clf"]
    row = words.transform([text])
    if row.nnz == 0:
        return []
    k = list(clf.classes_).index(category)
    n_word = len(words.vocabulary_)
    coefs = clf.coef_[k][:n_word]
    vocab = words.get_feature_names_out()
    idx = row.indices
    contrib = row.data * coefs[idx]
    top = idx[np.argsort(contrib)[::-1][:n_terms]]
    return [vocab[i] for i in top if coefs[i] > 0]
