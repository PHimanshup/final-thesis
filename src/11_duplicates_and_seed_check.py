"""
11_duplicates_and_seed_check.py
=================================
TWO QUESTIONS THIS SCRIPT ANSWERS:

  QUESTION 1: Why did cleaning remove about 69% of the UNSW-NB15 attack rows
  but only about 12% of the normal rows?
      Script 02 removes exact duplicate rows. This script reloads the RAW
      UNSW-NB15 files and counts duplicates separately for each attack
      category, so we can see WHICH kinds of attack are made of repeated
      rows, and print a few examples of what a "duplicate" actually looks
      like.

  QUESTION 2: Is the cross-dataset result (AUC below 0.5) a stable finding,
  or an accident of one training run or of the duplicate removal?
      We repeat the key experiment three times with different random seeds
      (neural networks start from random weights, so every run differs
      slightly), and we evaluate on UNSW-NB15 two ways:
          (a) with duplicates removed (what script 02 produced), and
          (b) with the raw rows kept (no duplicate removal).
      We also repeat the within-CICIDS2017 test (same 5 shared features).

HOW TO READ THE RESULT:
      - If (a) and (b) give similar AUC, the duplicate removal is not what
        drives your finding. One sentence in the report is enough.
      - If they differ, report BOTH and explain which one you trust and why.
      - The spread between seeds (min to max) is your run-to-run variation.
        Report the mean and the range, not a single number.

HOW TO RUN THIS:
      python3 src/11_duplicates_and_seed_check.py
      (Needs scripts 01, 02 and 10 in the src folder, and 01 and 02 already
       run. This retrains a model 3 times, so allow 30 to 60 minutes. Keep
       the laptop plugged in and awake.)
"""

import os
import sys
import gc
import importlib.util

import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR, RAW_UNSW_DIR, COMMON_FEATURE_MAP

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Seeds for the repeated runs. Each seed changes the neural network's
# starting weights; everything else stays identical.
SEEDS = [42, 43, 44]

MICROSECONDS_PER_SECOND = 1_000_000


def load_script(filename, module_name):
    """
    Our scripts start with numbers (01_, 02_ ...), which Python's normal
    'import' statement cannot handle, so we load them by file path. This
    lets us reuse the loader from script 02 and the training/evaluation
    functions from script 10 instead of copying them again.
    """
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SRC_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


unsw_loader = load_script("02_load_and_clean_unsw.py", "unsw_loader")
diagnostics = load_script("10_cross_dataset_diagnostics.py", "diagnostics")

UNSW_SHARED_COLS = list(COMMON_FEATURE_MAP.values())
CICIDS_SHARED_COLS = list(COMMON_FEATURE_MAP.keys())


def duplicate_report(raw):
    """
    Count exact duplicate rows (same definition as script 02: every one of
    the 49 columns identical) separately for each attack category.
    """
    category = raw["attack_cat"].fillna("Normal").astype(str).str.strip()
    is_duplicate = raw.duplicated(keep="first")

    table = (
        pd.DataFrame({"category": category, "is_duplicate": is_duplicate})
        .groupby("category")["is_duplicate"]
        .agg(total_rows="size", duplicate_rows="sum")
    )
    table["pct_duplicate"] = 100 * table["duplicate_rows"] / table["total_rows"]
    table = table.sort_values("pct_duplicate", ascending=False)

    print("\nDuplicate rows by UNSW-NB15 category (raw data):")
    print(table.to_string(float_format=lambda v: f"{v:,.1f}"))

    by_label = pd.DataFrame({"label": raw["label"], "is_duplicate": is_duplicate}) \
        .groupby("label")["is_duplicate"].agg(total_rows="size", duplicate_rows="sum")
    by_label["pct_duplicate"] = 100 * by_label["duplicate_rows"] / by_label["total_rows"]
    print("\nDuplicate rows by label (0 = normal, 1 = attack):")
    print(by_label.to_string(float_format=lambda v: f"{v:,.1f}"))

    # Show what a duplicated ATTACK row actually looks like
    example_cols = ["srcip", "sport", "dstip", "dsport", "proto", "state",
                    "dur", "sbytes", "stime", "attack_cat"]
    examples = raw[is_duplicate & (raw["label"] == 1)].head(5)[example_cols]
    print("\nFirst 5 duplicated attack rows (key columns only):")
    print(examples.to_string())

    return table


def shared_feature_matrix(df):
    """Pull out the shared features and the 0/1 attack label. Infinite or
    missing values are replaced by the column median, as in script 02."""
    X = df[UNSW_SHARED_COLS].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    y = (df["label"] != 0).astype(int).values
    return X.values, y


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # PART 1: duplicates in the RAW UNSW-NB15 files
    # ------------------------------------------------------------------
    print("Loading RAW UNSW-NB15 files (this takes a minute)...")
    raw = unsw_loader.load_all_unsw_files(RAW_UNSW_DIR)
    print(f"Raw rows: {len(raw):,}")

    table = duplicate_report(raw)
    table.to_csv(os.path.join(RESULTS_DIR, "unsw_duplicates_by_category.csv"))

    # Keep only what we need for the experiments, then free the big table
    X_unsw_raw, y_unsw_raw = shared_feature_matrix(raw)
    del raw
    gc.collect()

    cleaned = pd.read_pickle(os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl"))
    X_unsw_dedup, y_unsw_dedup = shared_feature_matrix(cleaned)
    del cleaned
    gc.collect()

    print(f"\nUNSW-NB15 raw (duplicates kept):    {len(y_unsw_raw):,} rows, "
          f"attack rate {y_unsw_raw.mean():.2%}")
    print(f"UNSW-NB15 deduplicated (script 02): {len(y_unsw_dedup):,} rows, "
          f"attack rate {y_unsw_dedup.mean():.2%}")

    # ------------------------------------------------------------------
    # PART 2: CICIDS2017 data, shared features, duration in seconds
    # ------------------------------------------------------------------
    cicids = pd.read_pickle(os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl"))
    cic_X = cicids[CICIDS_SHARED_COLS].astype(float).copy()
    cic_X["Flow Duration"] = cic_X["Flow Duration"] / MICROSECONDS_PER_SECOND
    cic_X = cic_X.values

    is_monday = cicids["source_file"].str.contains("Monday", case=False).values
    cic_train_X = cic_X[is_monday]
    cic_test_X = cic_X[~is_monday]
    cic_test_y = (cicids.loc[~is_monday, "Label"] != "BENIGN").astype(int).values
    del cicids
    gc.collect()

    # ------------------------------------------------------------------
    # PART 3: repeat the experiments with different seeds
    # ------------------------------------------------------------------
    rows = []
    for seed in SEEDS:
        print(f"\n===== Seed {seed} =====")
        keras.utils.set_random_seed(seed)   # seeds Python, NumPy and TensorFlow

        scaler, model, train_error = diagnostics.train_on(cic_train_X)

        results = [
            diagnostics.evaluate("A: within CICIDS2017", "CICIDS2017", "CICIDS2017",
                                 scaler, model, train_error, cic_test_X, cic_test_y),
            diagnostics.evaluate("B1: CICIDS2017 -> UNSW-NB15 (deduplicated)", "CICIDS2017",
                                 "UNSW-NB15 (deduplicated)", scaler, model, train_error,
                                 X_unsw_dedup, y_unsw_dedup),
            diagnostics.evaluate("B2: CICIDS2017 -> UNSW-NB15 (raw, duplicates kept)", "CICIDS2017",
                                 "UNSW-NB15 (raw)", scaler, model, train_error,
                                 X_unsw_raw, y_unsw_raw),
        ]
        for result in results:
            result["seed"] = seed
            rows.append(result)

    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(os.path.join(RESULTS_DIR, "seed_and_dedup_runs.csv"), index=False)

    summary = (
        all_runs.groupby("experiment")["auc_roc"]
        .agg(mean="mean", lowest="min", highest="max")
        .round(3)
    )
    print("\n\nAUC-ROC across seeds (mean, lowest, highest):")
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, "seed_and_dedup_summary.csv"))

    print("\nSaved: seed_and_dedup_runs.csv, seed_and_dedup_summary.csv, "
          "unsw_duplicates_by_category.csv in your results folder.")
