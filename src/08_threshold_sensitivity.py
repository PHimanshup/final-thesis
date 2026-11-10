"""
08_threshold_sensitivity.py
=============================
WHAT THIS SCRIPT DOES:
    Takes the trained Autoencoder and sweeps through many different
    anomaly-score thresholds (not just the 95th/99th percentile used in
    script 07), recalculating recall and false positive rate at each one.
    Saves a table and a plot showing how these two numbers trade off
    against each other as the threshold changes.

WHY THIS SCRIPT EXISTS:
    Your Literature Survey identifies a specific gap: most published IDS
    studies report only ONE performance figure at a single, fixed
    threshold, without showing how sensitive that figure is to the
    threshold choice. This script is the experiment that fills that gap
    for your own project — don't skip it, it's one of your more original
    contributions, not just a side analysis.

HOW TO RUN THIS:
    python src/08_threshold_sensitivity.py

    (Run scripts 04 and 06 first — this script needs their output.)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR


def reconstruction_error(model, X):
    reconstructed = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def threshold_sweep(scores, y_true, percentiles, reference_scores):
    """
    For each percentile in `percentiles`, set the anomaly threshold at
    that percentile of the REFERENCE scores (the autoencoder's error on
    BENIGN TRAINING data), then apply that threshold to the test-set
    `scores` and compute recall, precision and false positive rate.

    Using benign training error as the reference mirrors a real
    deployment, where you set the alarm level from normal traffic only,
    and it matches how script 07 sets its 95th and 99th percentile
    thresholds.
    """
    rows = []
    for p in percentiles:
        threshold = np.percentile(reference_scores, p)
        y_pred = (scores >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        rows.append({
            "percentile": p,
            "threshold_value": threshold,
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "false_positive_rate": false_positive_rate,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_binary.npy"))

    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if not os.path.exists(ae_path):
        raise FileNotFoundError("Autoencoder model not found. Run script 06 first.")
    autoencoder = keras.models.load_model(ae_path)

    print("Scoring benign training set and test set...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"))
    train_scores = reconstruction_error(autoencoder, X_train)
    test_scores = reconstruction_error(autoencoder, X_test)

    # Sweep every percentile from 80 to 99, in steps of 1, with the
    # thresholds taken from the benign training error distribution
    percentiles_to_test = np.arange(80, 100, 1)
    sweep_results = threshold_sweep(test_scores, y_test, percentiles_to_test, train_scores)

    sweep_path = os.path.join(RESULTS_DIR, "threshold_sensitivity_table.csv")
    sweep_results.to_csv(sweep_path, index=False)
    print(f"Saved threshold sweep table to: {sweep_path}")
    print(sweep_results)

    # --- Plot the detection-rate / false-alarm trade-off ---------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_results["percentile"], sweep_results["recall"],
            "o-", label="Recall (detection rate)", color="tab:blue")
    ax.plot(sweep_results["percentile"], sweep_results["false_positive_rate"],
            "s-", label="False Positive Rate", color="tab:red")
    ax.set_xlabel("Threshold (percentile of benign training reconstruction error)")
    ax.set_ylabel("Rate")
    ax.set_title("Autoencoder — Threshold Sensitivity\n(Detection Rate vs False Alarm Trade-off)")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "threshold_sensitivity_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved threshold sensitivity plot to: {plot_path}")
