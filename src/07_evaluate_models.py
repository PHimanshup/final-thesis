"""
07_evaluate_models.py
=======================
WHAT THIS SCRIPT DOES:
    Loads both trained models (all three Isolation Forests and the
    Autoencoder) plus the held-out test set, and scores each model using
    the five metrics named in your Aim: precision, recall, F1-score,
    AUC-ROC, and false positive rate. Saves a results table (CSV) and an
    ROC curve plot — both go straight into your Testing section.

WHAT EACH METRIC MEANS, IN PLAIN TERMS:
    - Precision: of everything the model flagged as an attack, what
      fraction actually was an attack? Low precision = lots of false alarms.
    - Recall: of all the real attacks in the test set, what fraction did
      the model catch? Low recall = lots of missed attacks.
    - F1-score: a single number that balances precision and recall.
    - AUC-ROC: summarises how well the model separates attacks from
      benign traffic across EVERY possible threshold, not just one. 1.0 is
      perfect, 0.5 is random guessing.
    - False Positive Rate: of all the genuinely benign traffic, what
      fraction got wrongly flagged as an attack? This is the "false alarm"
      cost that a security analyst would actually have to deal with.

HOW TO RUN THIS:
    python src/07_evaluate_models.py

    (Run scripts 04, 05, and 06 first — this script needs their output.)
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, ISOLATION_FOREST_CONTAMINATIONS


def isolation_forest_scores(model, X):
    """
    sklearn's score_samples() returns HIGHER values for MORE NORMAL points
    (the opposite of what we want). We flip the sign so that, consistent
    with the autoencoder's reconstruction error, a HIGHER score here means
    MORE anomalous — this makes the two models directly comparable.
    """
    return -model.score_samples(X)


def reconstruction_error(model, X):
    """
    Feed data through the autoencoder, then measure how different the
    rebuilt output is from the original input (mean squared error per
    row). This is our anomaly score for the autoencoder.
    """
    reconstructed = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


def compute_metrics(y_true, y_pred, scores, model_name, threshold_description):
    """
    Given true labels, predicted labels (0/1), and continuous anomaly
    scores, compute all five metrics named in the Aim and return them as
    a dictionary (one row of our final results table).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "model": model_name,
        "threshold": threshold_description,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, scores),
        "false_positive_rate": false_positive_rate,
    }


if __name__ == "__main__":
    # --- Load test data -----------------------------------------------------
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_binary.npy"))
    print(f"Loaded test set: X_test={X_test.shape}, attack rate={y_test.mean():.2%}")

    all_results = []
    all_scores = {}  # keep raw scores around so we can plot ROC curves below

    # --- Evaluate each Isolation Forest model --------------------------------
    for contamination in ISOLATION_FOREST_CONTAMINATIONS:
        model_path = os.path.join(MODELS_DIR, f"isolation_forest_c{contamination}.joblib")
        if not os.path.exists(model_path):
            print(f"Skipping contamination={contamination}: model file not found. Run script 05 first.")
            continue

        model = joblib.load(model_path)
        scores = isolation_forest_scores(model, X_test)

        # Isolation Forest's own .predict() gives -1 (anomaly) / 1 (normal)
        # using the contamination value it was trained with — this is its
        # "native" decision, separate from the percentile-threshold
        # approach we use for the autoencoder in script 08
        native_prediction = (model.predict(X_test) == -1).astype(int)

        model_name = f"Isolation Forest (contamination={contamination})"
        result = compute_metrics(y_test, native_prediction, scores, model_name, "native")
        all_results.append(result)
        all_scores[model_name] = scores

        print(f"\n{model_name}")
        for key, value in result.items():
            if key not in ["model", "threshold"]:
                print(f"  {key}: {value:.4f}")

    # --- Evaluate the Autoencoder --------------------------------------------
    ae_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    if not os.path.exists(ae_path):
        print("Autoencoder model not found. Run script 06 first.")
    else:
        autoencoder = keras.models.load_model(ae_path)

        # Threshold is set from the BENIGN TRAINING data's error distribution
        # (not the test set) — this mirrors how you would set a threshold in
        # a real deployment, where you don't have labelled attack data to
        # tune against.
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"))
        train_error = reconstruction_error(autoencoder, X_train)
        test_scores = reconstruction_error(autoencoder, X_test)
        all_scores["Autoencoder"] = test_scores

        for percentile in [95, 99]:
            threshold = np.percentile(train_error, percentile)
            predictions = (test_scores >= threshold).astype(int)

            model_name = f"Autoencoder ({percentile}th percentile threshold)"
            result = compute_metrics(y_test, predictions, test_scores, model_name, f"{percentile}th pct")
            all_results.append(result)

            print(f"\n{model_name}")
            for key, value in result.items():
                if key not in ["model", "threshold"]:
                    print(f"  {key}: {value:.4f}")

    # --- Save results table ----------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(RESULTS_DIR, "model_evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results table to: {results_path}")

    # --- Plot ROC curves for both models side by side --------------------------
    plt.figure(figsize=(8, 6))
    for model_name, scores in all_scores.items():
        # Skip the two duplicate autoencoder entries at different thresholds —
        # the ROC curve itself doesn't depend on which threshold we chose,
        # so we only need to plot it once
        if model_name in all_scores:
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc = roc_auc_score(y_test, scores)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — CICIDS2017 Test Set")
    plt.legend(fontsize=8)
    plt.tight_layout()

    roc_path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_path}")
