"""
15_pca_experiment.py
====================
WHAT THIS SCRIPT DOES:
    Adds PCA (Principal Component Analysis) as a third unsupervised detector,
    as listed in the project proposal.

    PCA is fitted ONLY on the benign Monday training data. It squeezes each
    flow down to a few "components" and then rebuilds it. Normal traffic
    rebuilds well (small error). Traffic that looks different rebuilds badly
    (large error). That reconstruction error is the anomaly score, the same
    idea as the autoencoder but with a simple linear method.

    We report the same measures as the other models (precision, recall, F1,
    AUC-ROC, false positive rate) at the 95th and 99th percentile of the
    BENIGN TRAINING error, for several numbers of components.

HOW TO RUN (needs scripts 01 and 04 to have been run first):
    python3 src/15_pca_experiment.py
    (Takes a minute or two.)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR

# How many components to try. 78 features go down to this many.
COMPONENT_OPTIONS = [5, 10, 20, 30]
THRESHOLD_PERCENTILES = [95, 99]
CHUNK = 200_000   # score the big test set in pieces so the Mac does not run out of memory


def reconstruction_error(pca, X):
    """Mean squared difference between each row and its PCA rebuild."""
    errors = np.empty(len(X))
    for start in range(0, len(X), CHUNK):
        chunk = X[start:start + CHUNK]
        rebuilt = pca.inverse_transform(pca.transform(chunk))
        errors[start:start + CHUNK] = np.mean((chunk - rebuilt) ** 2, axis=1)
    return errors


def evaluate(scores, y_true, threshold):
    """Precision, recall, F1 and false positive rate at one threshold."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return precision, recall, f1, fpr


def run_pca(X_train, X_test, y_test, n_components):
    """Fit PCA on benign training data only, score the test data, return result rows."""
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(X_train)
    explained = float(pca.explained_variance_ratio_.sum())

    train_error = reconstruction_error(pca, X_train)
    test_error = reconstruction_error(pca, X_test)
    auc = roc_auc_score(y_test, test_error)

    rows = []
    for pct in THRESHOLD_PERCENTILES:
        threshold = np.percentile(train_error, pct)   # from benign TRAINING error only
        precision, recall, f1, fpr = evaluate(test_error, y_test, threshold)
        rows.append({
            "model": f"PCA ({n_components} components)",
            "variance_explained": round(explained, 4),
            "threshold": f"{pct}th pct",
            "precision": precision, "recall": recall, "f1_score": f1,
            "auc_roc": auc, "false_positive_rate": fpr,
        })
    print(f"  {n_components:>2} components keep {explained:.1%} of the variance -> AUC {auc:.3f}")
    return rows


if __name__ == "__main__":
    needed = ["X_train_scaled.npy", "X_test_scaled.npy", "y_test_binary.npy"]
    for name in needed:
        if not os.path.exists(os.path.join(PROCESSED_DIR, name)):
            raise FileNotFoundError(f"{name} not found. Run 04_split_and_scale.py first.")

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_binary.npy"))
    print(f"Train (benign only): {X_train.shape}   Test: {X_test.shape}, attack rate {y_test.mean():.2%}")

    all_rows = []
    for n in COMPONENT_OPTIONS:
        all_rows.extend(run_pca(X_train, X_test, y_test, n))

    results = pd.DataFrame(all_rows)
    out_path = os.path.join(RESULTS_DIR, "pca_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(results.round(3).to_string(index=False))
