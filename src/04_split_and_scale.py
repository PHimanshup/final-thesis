"""
04_split_and_scale.py
=======================
WHAT THIS SCRIPT DOES:
    1. Loads the cleaned CICIDS2017 data
    2. Splits it into a TRAINING set (Monday only — 100% benign traffic)
       and a TEST set (Tuesday to Friday — a mix of benign and attack
       traffic), exactly as described in your Methodology
    3. Fits a feature scaler (StandardScaler) using ONLY the benign
       training data, then applies it to both the training and test sets
    4. Saves the resulting arrays and the fitted scaler, so later scripts
       can load them directly instead of repeating this work

WHY WE FIT THE SCALER ONLY ON BENIGN TRAINING DATA:
    If we fit the scaler on the test set too (which contains attacks), the
    scaler's idea of "normal range" would be influenced by attack traffic
    statistics. That would leak information from the test set into our
    preprocessing, making our later results look better than they really
    are. Fitting only on benign training data avoids this leak.

HOW TO RUN THIS:
    python src/04_split_and_scale.py

    (Run 01_load_and_clean_cicids.py first — this script needs its output.)
"""

import os
import sys

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODELS_DIR


if __name__ == "__main__":
    # --- Load the cleaned data from script 01 -----------------------------
    cicids_path = os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl")
    if not os.path.exists(cicids_path):
        raise FileNotFoundError(
            "cicids2017_clean.pkl not found. Run 01_load_and_clean_cicids.py first."
        )
    df = pd.read_pickle(cicids_path)

    # --- Split by day: Monday = training, Tue-Fri = test -------------------
    # CICIDS2017 filenames typically look like "Monday-WorkingHours.pcap_ISCX.csv"
    # so we match on whether "Monday" appears in the filename we saved in
    # script 01's "source_file" column.
    is_monday = df["source_file"].str.contains("Monday", case=False)

    train_df = df[is_monday].copy()
    test_df = df[~is_monday].copy()

    print(f"Training rows (Monday, should be all benign): {len(train_df):,}")
    print(f"Test rows (Tuesday-Friday, mixed):             {len(test_df):,}")

    # Sanity check: the training set should be 100% benign. If this fails,
    # either the filenames don't contain "Monday" as expected, or the
    # dataset you downloaded is structured differently — check before
    # continuing, because the whole unsupervised approach depends on this.
    non_benign_in_train = (train_df["Label"] != "BENIGN").sum()
    if non_benign_in_train > 0:
        print(
            f"\nWARNING: {non_benign_in_train} non-benign rows found in the "
            "'Monday' training set. Check your source_file names — this "
            "script assumes Monday's capture is 100% benign, per your "
            "Methodology."
        )
    else:
        print("Confirmed: training set is 100% benign, as expected.")

    # --- Separate features from labels --------------------------------------
    # We drop 'Label' (the attack type) and 'source_file' (our own tracking
    # column) because these are not real network traffic features — the
    # models should never see them during training.
    columns_to_drop = ["Label", "source_file"]
    feature_columns = [c for c in df.columns if c not in columns_to_drop]

    X_train = train_df[feature_columns].values
    X_test = test_df[feature_columns].values

    # y_test_binary: 1 = attack, 0 = benign. This is ONLY used for
    # evaluation later (scoring the models) — the models never see this
    # during training, which is what makes this an unsupervised approach.
    y_test_binary = (test_df["Label"] != "BENIGN").astype(int).values

    print(f"\nFeature matrix shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Test set attack rate: {y_test_binary.mean():.2%}")

    # --- Fit the scaler on BENIGN TRAINING DATA ONLY ------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only, never fit, on test data

    # --- Save everything for the next scripts to use ------------------------
    np.save(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, "y_test_binary.npy"), y_test_binary)

    # joblib is the standard way to save fitted scikit-learn objects
    # (like our scaler) so we can reuse the exact same fitted scaler later
    # without re-fitting it
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.joblib"))

    print(f"\nSaved scaled arrays to {PROCESSED_DIR}")
    print(f"Saved fitted scaler to {MODELS_DIR}")
