"""
12_per_attack_type.py
======================
WHAT THIS SCRIPT DOES:
    Your overall results (script 07) average every attack type together. That
    hides which attacks the models catch and which they miss. This script
    breaks the CICIDS2017 test results down by attack type.

    For each attack type it reports:
        - recall: what fraction of that attack's flows were flagged
        - AUC vs benign: how well the model's anomaly score separates THAT
          attack from benign traffic (1.0 = perfectly, 0.5 = like guessing)
    for both the Autoencoder (95th percentile threshold) and the Isolation
    Forest (contamination 0.10, its own built-in cutoff).

    It also reports the false positive rate on benign traffic, so you can see
    what the recall figures cost in false alarms.

WHY THIS MATTERS FOR YOUR PROJECT:
    Your first research question asks how dataset characteristics such as
    class imbalance affect the models. Some attack types in CICIDS2017 have
    only a handful of flows (for example Heartbleed or SQL Injection). This
    table shows how many rows each attack really has, so you can say which
    results are reliable and which rest on very few rows.

HOW TO RUN THIS:
    python3 src/12_per_attack_type.py
    (Run scripts 01, 04, 05 and 06 first. This scores about 2.3 million rows,
     so allow several minutes.)
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, ISOLATION_FOREST_CONTAMINATIONS

# Attack types with fewer rows than this give very unstable AUC figures
SMALL_SAMPLE_LIMIT = 100


def clean_label(series):
    """
    Some CICIDS2017 labels contain a damaged dash (for example
    'Web Attack ? Brute Force'). Replace any non-ASCII character with a plain
    dash so each attack type has one clean, readable name.
    """
    return series.astype(str).str.replace(r"[^\x00-\x7F]+", "-", regex=True).str.strip()


def reconstruction_error(model, X):
    """Autoencoder anomaly score: mean squared rebuild error per row."""
    rebuilt = model.predict(X, verbose=0, batch_size=4096)
    return np.mean(np.square(X - rebuilt), axis=1)


if __name__ == "__main__":
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"))

    # The test rows are every non-Monday row, in the same order script 04 used,
    # so the labels line up with X_test row by row.
    cicids = pd.read_pickle(os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl"))
    is_monday = cicids["source_file"].str.contains("Monday", case=False)
    test_labels = clean_label(cicids.loc[~is_monday, "Label"]).values
    del cicids

    if len(test_labels) != len(X_test):
        raise ValueError(
            f"Label count ({len(test_labels)}) does not match test rows ({len(X_test)}). "
            "Re-run scripts 01 and 04 so they use the same data."
        )

    # --- Autoencoder scores and its flags at the 95th percentile threshold ----
    autoencoder = keras.models.load_model(os.path.join(MODELS_DIR, "autoencoder.keras"))
    threshold = np.percentile(reconstruction_error(autoencoder, X_train), 95)
    ae_scores = reconstruction_error(autoencoder, X_test)
    ae_flagged = ae_scores >= threshold

    # --- Isolation Forest scores and its own built-in flags -----------------------
    contamination = ISOLATION_FOREST_CONTAMINATIONS[1]  # 0.10
    iso_path = os.path.join(MODELS_DIR, f"isolation_forest_c{contamination}.joblib")
    iso = joblib.load(iso_path)
    iso_scores = -iso.score_samples(X_test)       # flipped so higher = more anomalous
    iso_flagged = iso.predict(X_test) == -1

    is_benign = test_labels == "BENIGN"

    rows = [{
        "traffic_type": "BENIGN (false positive rate)",
        "n_rows": int(is_benign.sum()),
        "autoencoder_flagged": ae_flagged[is_benign].mean(),
        "autoencoder_auc_vs_benign": np.nan,
        "iforest_flagged": iso_flagged[is_benign].mean(),
        "iforest_auc_vs_benign": np.nan,
        "note": "share of normal traffic wrongly flagged",
    }]

    for attack in sorted(set(test_labels) - {"BENIGN"}):
        this_attack = test_labels == attack
        attack_or_benign = this_attack | is_benign
        y_true = this_attack[attack_or_benign].astype(int)

        rows.append({
            "traffic_type": attack,
            "n_rows": int(this_attack.sum()),
            "autoencoder_flagged": ae_flagged[this_attack].mean(),
            "autoencoder_auc_vs_benign": roc_auc_score(y_true, ae_scores[attack_or_benign]),
            "iforest_flagged": iso_flagged[this_attack].mean(),
            "iforest_auc_vs_benign": roc_auc_score(y_true, iso_scores[attack_or_benign]),
            "note": "very few rows, treat with caution" if this_attack.sum() < SMALL_SAMPLE_LIMIT else "",
        })

    table = pd.DataFrame(rows)
    benign_row = table.iloc[:1]
    attack_rows = table.iloc[1:].sort_values("n_rows", ascending=False)
    table = pd.concat([benign_row, attack_rows], ignore_index=True)

    print("\nResults by traffic type (CICIDS2017 test set, Tuesday to Friday)")
    print("'flagged' = share of that traffic the model marked as an attack")
    print("(for attacks this is recall; for benign it is the false positive rate)\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out_path = os.path.join(RESULTS_DIR, "per_attack_type_results.csv")
    table.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")
