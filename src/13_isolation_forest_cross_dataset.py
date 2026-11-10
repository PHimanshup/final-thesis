"""
13_isolation_forest_cross_dataset.py
======================================
WHAT THIS SCRIPT DOES:
    Your proposal promised to test the trained models on UNSW-NB15 without
    retraining. So far only the Autoencoder was tested across datasets. This
    script runs the same cross-dataset test for the Isolation Forest, so both
    models are compared fairly.

    Exactly like the Autoencoder test in scripts 10 and 11:
        - only the 5 features both datasets share are used
        - CICIDS2017 flow duration is converted from microseconds to seconds
        - the model and scaler are trained on CICIDS2017 Monday (benign) only
        - the model is then applied to UNSW-NB15 with NO retraining
        - the run is repeated with three random seeds

    Two settings are reported for each run:
        - AUC-ROC (how well the anomaly score ranks attacks above normal)
        - recall and false positive rate at the model's own built-in cutoff

HOW TO RUN THIS:
    python3 src/13_isolation_forest_cross_dataset.py
    (Run scripts 01 and 02 first. It takes a few minutes.)
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, COMMON_FEATURE_MAP

SEEDS = [42, 43, 44]
MAX_TRAIN_ROWS = 500_000
MICROSECONDS_PER_SECOND = 1_000_000
CONTAMINATION = 0.10   # the middle value from your proposal


def evaluate(name, model, scaler, X, y, seed):
    """Score data with a trained Isolation Forest and report AUC, plus recall
    and false positive rate at the model's own built-in cutoff."""
    X_scaled = scaler.transform(X)
    scores = -model.score_samples(X_scaled)          # higher = more anomalous
    flagged = model.predict(X_scaled) == -1

    row = {
        "experiment": name,
        "seed": seed,
        "n_rows": len(y),
        "attack_rate": float(np.mean(y)),
        "auc_roc": roc_auc_score(y, scores),
        "recall": float(flagged[y == 1].mean()),
        "false_positive_rate": float(flagged[y == 0].mean()),
    }
    print(f"  {name}: AUC={row['auc_roc']:.3f}  recall={row['recall']:.3f}  "
          f"FPR={row['false_positive_rate']:.3f}")
    return row


if __name__ == "__main__":
    cicids = pd.read_pickle(os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl"))
    unsw = pd.read_pickle(os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl"))

    cic_cols = list(COMMON_FEATURE_MAP.keys())
    unsw_cols = list(COMMON_FEATURE_MAP.values())

    # CICIDS2017: shared features, duration converted to seconds
    cic_X = cicids[cic_cols].astype(float).copy()
    cic_X["Flow Duration"] = cic_X["Flow Duration"] / MICROSECONDS_PER_SECOND
    cic_X = cic_X.values
    is_monday = cicids["source_file"].str.contains("Monday", case=False).values
    cic_train_X = cic_X[is_monday]
    cic_test_X = cic_X[~is_monday]
    cic_test_y = (cicids.loc[~is_monday, "Label"] != "BENIGN").astype(int).values

    # UNSW-NB15 (deduplicated, as produced by script 02): shared features
    unsw_X = unsw[unsw_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    unsw_X = unsw_X.fillna(unsw_X.median()).values
    unsw_y = (unsw["label"] != 0).astype(int).values

    rows = []
    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")
        rng = np.random.default_rng(seed)
        train_X = cic_train_X
        if len(train_X) > MAX_TRAIN_ROWS:
            train_X = train_X[rng.choice(len(train_X), MAX_TRAIN_ROWS, replace=False)]

        scaler = StandardScaler().fit(train_X)          # fitted on benign training data only
        model = IsolationForest(
            n_estimators=200, contamination=CONTAMINATION,
            random_state=seed, n_jobs=-1,
        ).fit(scaler.transform(train_X))

        rows.append(evaluate("A-IF: within CICIDS2017", model, scaler, cic_test_X, cic_test_y, seed))
        rows.append(evaluate("B-IF: CICIDS2017 -> UNSW-NB15", model, scaler, unsw_X, unsw_y, seed))

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(RESULTS_DIR, "isolation_forest_cross_dataset_runs.csv"), index=False)

    summary = runs.groupby("experiment")["auc_roc"].agg(mean="mean", lowest="min", highest="max").round(3)
    print("\nIsolation Forest AUC-ROC across seeds:")
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, "isolation_forest_cross_dataset_summary.csv"))
    print("\nSaved results to your results folder.")
    print("Compare these numbers with the Autoencoder's A and B results from script 11.")
