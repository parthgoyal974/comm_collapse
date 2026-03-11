"""
baselines/tfidf_lr.py
=====================
Classical ML baseline: TF-IDF + Logistic Regression

Serves as the lower-bound comparison in the results table.

Run:
    python baselines/tfidf_lr.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score, classification_report
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_processed
from tokenize_utils import serialize_window

METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def window_to_text(window: dict) -> str:
    """Flatten all turn texts from a window into a single string."""
    return " ".join(t["text"] for t in window["turns"])


def run_tfidf_baseline():
    print("\n" + "="*55)
    print("  BASELINE: TF-IDF + Logistic Regression")
    print("="*55)

    # Load data
    print("\nLoading data...")
    train_windows = load_processed("train")
    dev_windows   = load_processed("dev")
    test_windows  = load_processed("test")

    # Featurize
    print("Building TF-IDF features...")
    X_train = [window_to_text(w) for w in train_windows]
    y_train = [w["label"] for w in train_windows]
    X_dev   = [window_to_text(w) for w in dev_windows]
    y_dev   = [w["label"] for w in dev_windows]
    X_test  = [window_to_text(w) for w in test_windows]
    y_test  = [w["label"] for w in test_windows]

    # Pipeline: TF-IDF (unigrams + bigrams) + LR
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    # Evaluate on dev
    dev_probs = pipeline.predict_proba(X_dev)[:, 1]
    dev_preds = pipeline.predict(X_dev)
    dev_f1 = f1_score(y_dev, dev_preds, average="macro", zero_division=0)
    print(f"\nDev F1 Macro: {dev_f1:.4f}")

    # Evaluate on test
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = pipeline.predict(X_test)

    test_f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_f1_pos   = f1_score(y_test, test_preds, average="binary", zero_division=0)
    try:
        test_auroc = roc_auc_score(y_test, test_probs)
        test_auprc = average_precision_score(y_test, test_probs)
    except Exception:
        test_auroc = test_auprc = 0.0

    metrics = {
        "model": "TF-IDF + LogisticRegression",
        "f1_macro":   test_f1_macro,
        "f1_positive": test_f1_pos,
        "auroc":       test_auroc,
        "auprc":       test_auprc,
    }

    print("\n=== TEST RESULTS ===")
    print(f"  F1 Macro:   {test_f1_macro:.4f}")
    print(f"  F1 Pos:     {test_f1_pos:.4f}")
    print(f"  AUROC:      {test_auroc:.4f}")
    print(f"  AUPRC:      {test_auprc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds,
                                target_names=["Normal", "Breakdown"], zero_division=0))

    out = METRICS_DIR / "baseline_tfidf_lr_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {out}")
    return metrics


if __name__ == "__main__":
    run_tfidf_baseline()
