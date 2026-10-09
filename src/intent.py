"""Reusable pieces for the Banking77 intent-classification project.

Everything the notebook needs lives here so the analysis stays readable and
the logic can be tested or reused outside the notebook.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 42


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the official Banking77 train/test splits."""
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def _word_char_features() -> object:
    """Word 1-2 grams plus character 2-5 grams (robust to typos and inflections)."""
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
    )


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    """Candidate models, simplest first, each with a small hyperparameter grid."""
    return {
        "Bag-of-words + Naive Bayes": (
            Pipeline([("vec", CountVectorizer()), ("clf", MultinomialNB())]),
            {"clf__alpha": [0.03, 0.1, 0.3, 1.0]},
        ),
        "TF-IDF words + Logistic Regression": (
            Pipeline([
                ("vec", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("clf", LogisticRegression(max_iter=3000)),
            ]),
            {"clf__C": [5, 20, 50]},
        ),
        "TF-IDF words+chars + Linear SVM": (
            Pipeline([("vec", _word_char_features()), ("clf", LinearSVC())]),
            {"clf__C": [0.2, 0.5, 1.0]},
        ),
        "TF-IDF words+chars + Logistic Regression": (
            Pipeline([("vec", _word_char_features()), ("clf", LogisticRegression(max_iter=3000))]),
            {"clf__C": [10, 30, 100]},
        ),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED):
    """Percentile bootstrap confidence interval for accuracy."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    n = len(correct)
    stats = correct[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return np.quantile(stats, [alpha / 2, 1 - alpha / 2])


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: do two classifiers differ on the same test items?"""
    from scipy.stats import binomtest

    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_only = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    b_only = int(((pred_b == y_true) & (pred_a != y_true)).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_right_b_wrong": a_only, "b_right_a_wrong": b_only, "p_value": p}


def routing_table(y_true, proba: np.ndarray, classes, thresholds=None) -> pd.DataFrame:
    """Coverage vs. accuracy if only predictions above a confidence threshold are automated.

    Queries below the threshold are sent to a human agent instead.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 0.96, 0.05), 2)
    y_true = np.asarray(y_true)
    conf = proba.max(axis=1)
    pred = np.asarray(classes)[proba.argmax(axis=1)]
    rows = []
    for t in thresholds:
        auto = conf >= t
        rows.append({
            "threshold": t,
            "automated_share": auto.mean(),
            "automated_accuracy": (pred[auto] == y_true[auto]).mean() if auto.any() else np.nan,
            "to_human_share": 1 - auto.mean(),
        })
    return pd.DataFrame(rows)


def top_confusions(y_true, y_pred, k: int = 10) -> pd.DataFrame:
    """Most frequent (true, predicted) error pairs."""
    df = pd.DataFrame({"true": y_true, "predicted": y_pred})
    errs = df[df.true != df.predicted]
    return (errs.value_counts().rename("count").reset_index().head(k))
