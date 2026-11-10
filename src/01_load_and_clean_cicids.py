"""
01_load_and_clean_cicids.py
============================
WHAT THIS SCRIPT DOES:
    1. Finds every CSV file in data/raw/cicids2017/ (CICIDS2017 comes as one
       CSV per day of the week — Monday, Tuesday, etc.)
    2. Loads them all and stacks them into one big table (DataFrame)
    3. Cleans that table: fixes infinite values, fills in missing values,
       removes duplicate rows, and tidies up the attack label text
    4. Saves the cleaned result to data/processed/ so every later script
       can just load the clean version instead of redoing this work

WHY WE CLEAN THE DATA THIS WAY:
    CICIDS2017 is known in the literature (Engelen et al., 2021) to contain
    infinite values, duplicate rows, and messy label text. If we don't fix
    these, they will quietly break the machine learning models later, or
    worse, silently distort the results without throwing an error.

HOW TO RUN THIS:
    Open a terminal in VS Code (Terminal > New Terminal), make sure your
    virtual environment is active, then run:
        python src/01_load_and_clean_cicids.py
"""

import os
import sys
import glob

import pandas as pd
import numpy as np

# This lets the script find config.py, which lives one folder up (in the
# project root), not inside src/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_CICIDS_DIR, PROCESSED_DIR


def load_all_cicids_files(folder):
    """
    Find every .csv file in `folder`, load each one, and stack them into
    a single table. We also remember which file each row came from (the
    day of the week), because later we need to know which rows are from
    "Monday" specifically, to use as our benign-only training set.
    """
    csv_paths = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {folder}\n"
            "Did you copy your CICIDS2017 day-files into data/raw/cicids2017/ ?"
        )

    all_days = []
    for path in csv_paths:
        day_df = pd.read_csv(path, low_memory=False)

        # CICIDS2017's column headers have stray leading/trailing spaces
        # in the original files (e.g. " Flow Duration" instead of
        # "Flow Duration") — this fixes that so column names match what
        # we expect everywhere else in the project.
        day_df.columns = day_df.columns.str.strip()

        # Remember which file this row came from, so we can filter by day later
        day_df["source_file"] = os.path.basename(path)

        print(f"  Loaded {os.path.basename(path):45s} -> {day_df.shape[0]:,} rows")
        all_days.append(day_df)

    combined = pd.concat(all_days, ignore_index=True)
    return combined


def clean_cicids(df):
    """
    Clean the raw combined CICIDS2017 table. Four specific problems are
    fixed here, each one explained as we go.
    """
    df = df.copy()
    rows_before = len(df)

    # --- Problem 1: infinite values -------------------------------------
    # Some flow-rate features (like "Flow Bytes/s") can be calculated as
    # a division by a very small or zero duration, producing "inf".
    # Machine learning models cannot handle infinite values, so we first
    # turn them into NaN (missing), then fill those with the column's
    # median value in the next step.
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # --- Problem 2: missing values ---------------------------------------
    # We fill any missing numeric value with that column's median (the
    # "middle" value), rather than the mean, because network traffic
    # features are often skewed (a few very large values), and the median
    # is less distorted by those extreme values than the mean would be.
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # --- Problem 3: messy label text --------------------------------------
    # The "Label" column sometimes has inconsistent spacing or casing
    # between files (e.g. "BENIGN" vs " BENIGN"). Stripping whitespace
    # prevents the same label being accidentally treated as two different
    # classes.
    df["Label"] = df["Label"].astype(str).str.strip()

    # --- Problem 4: duplicate rows -----------------------------------------
    # Exact duplicate rows don't add any new information and can bias the
    # model towards whatever pattern happens to be duplicated. We remove
    # them. "source_file" is deliberately left out of the duplicate check, so
    # an identical flow that appears on two different days counts as a
    # duplicate and only the first copy is kept. A side effect is that test
    # flows identical to training flows are removed, which avoids the model
    # being tested on traffic it has already seen. Mention this in your
    # Methodology.
    columns_to_check = [c for c in df.columns if c != "source_file"]
    df = df.drop_duplicates(subset=columns_to_check)

    rows_after = len(df)
    print(f"\nCleaning summary:")
    print(f"  Rows before cleaning: {rows_before:,}")
    print(f"  Rows after cleaning:  {rows_after:,}")
    print(f"  Rows removed:         {rows_before - rows_after:,}")

    return df


if __name__ == "__main__":
    print("Loading CICIDS2017 files...")
    raw_data = load_all_cicids_files(RAW_CICIDS_DIR)
    print(f"\nTotal combined shape before cleaning: {raw_data.shape}")

    print("\nCleaning data...")
    clean_data = clean_cicids(raw_data)

    # Save the cleaned data so later scripts don't need to repeat this work.
    # We use pandas' own .pkl format (pickle) because it's much faster to
    # save/load than CSV for large tables, and it preserves data types
    # exactly (CSV can silently change number formats).
    output_path = os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl")
    clean_data.to_pickle(output_path)
    print(f"\nSaved cleaned CICIDS2017 data to: {output_path}")

    print("\nLabel distribution after cleaning:")
    print(clean_data["Label"].value_counts())
