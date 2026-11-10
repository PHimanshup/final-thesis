"""
03_eda_and_suitability.py
===========================
WHAT THIS SCRIPT DOES:
    1. Loads the cleaned CICIDS2017 data (script 01) and the cleaned
       UNSW-NB15 data (script 02)
    2. Produces basic exploratory plots (class balance) and saves them as
       image files you can put directly into your report
    3. Runs a "dataset suitability report" — a set of defined checks that
       tells you, with real numbers, whether the dataset has the problems
       your supervisor asked you to check for (class imbalance, missing
       data, duplication)

WHY THIS SCRIPT EXISTS (important — read this):
    Your supervisor's feedback said: "ensure you validate your dataset
    against criteria for suitability and where it does not meet your
    needs." This script is the direct, concrete answer to that comment.
    The numbers this script prints are what you should quote, with real
    values, in your Methodology section — not a vague sentence like
    "the dataset has some limitations."

HOW TO RUN THIS:
    python src/03_eda_and_suitability.py

    (Run 01_load_and_clean_cicids.py and 02_load_and_clean_unsw.py first —
     this script needs their output.)
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, RESULTS_DIR


def plot_class_distribution(df, label_col, title, save_path):
    """
    A simple bar chart showing how many rows belong to each class
    (BENIGN vs each attack type). This is the first thing any reader
    of your dissertation will want to see, because it tells them
    immediately how imbalanced the data is.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    df[label_col].value_counts().plot(kind="bar", ax=ax)
    ax.set_yscale("log")  # log scale, so rare classes are visible next to huge ones
    ax.set_title(title)
    ax.set_ylabel("Number of flows (log scale)")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved chart: {save_path}")


def suitability_report(df, label_col, dataset_name):
    """
    This function defines FOUR specific, measurable criteria and checks
    the dataset against each one. These are the "criteria for suitability"
    your supervisor is asking you to define and apply, not leave implicit.

    Criterion 1 — Class imbalance ratio:
        How many times bigger is the largest class than the smallest?
        A very high ratio (e.g. 1000:1) means the model will barely see
        examples of rare attack types during evaluation, which affects
        how much we can trust recall figures for those rare classes.

    Criterion 2 — Missing data rate:
        What proportion of values were missing or infinite before
        cleaning? A high rate here would mean our median-imputation step
        is doing a lot of "guessing" rather than using real data.

    Criterion 3 — Duplicate row rate:
        What proportion of rows were exact duplicates? High duplication
        can make the model look artificially confident, because it's
        essentially seeing the same example many times.

    Criterion 4 — Dataset age / realism (not something code can measure):
        This one you answer with a sentence in your report, not a
        number — both CICIDS2017 (2017) and UNSW-NB15 (2015) are
        lab-generated, synthetic traffic, not live production traffic.
        State this explicitly as a known limitation in your Methodology.
    """
    print(f"\n{'='*60}")
    print(f"SUITABILITY REPORT: {dataset_name}")
    print(f"{'='*60}")

    total_rows = len(df)
    print(f"Total rows:    {total_rows:,}")
    print(f"Total columns: {df.shape[1]}")

    # Criterion 1: class imbalance
    class_counts = df[label_col].value_counts()
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\nCriterion 1 — Class imbalance ratio: {imbalance_ratio:,.1f} : 1")
    print(f"  (largest class = {class_counts.idxmax()!r} with {class_counts.max():,} rows)")
    print(f"  (smallest class = {class_counts.idxmin()!r} with {class_counts.min():,} rows)")

    # Criterion 2: missing values. NOTE: this script only sees the CLEANED data,
    # so this figure will be near 0% by design. The before/after cleaning row
    # counts printed by scripts 01 and 02 are the real evidence of data quality.
    numeric_cols = df.select_dtypes(include="number").columns
    missing_rate = df[numeric_cols].isna().mean().mean() * 100
    print(f"\nCriterion 2 — Mean missing-value rate across numeric columns: {missing_rate:.3f}%")

    # Criterion 3: duplicate rows remaining (should be close to 0% since
    # script 01/02 already removed duplicates — this check here confirms
    # the cleaning worked)
    duplicate_rate = df.duplicated().mean() * 100
    print(f"\nCriterion 3 — Remaining duplicate row rate: {duplicate_rate:.3f}%")

    print(f"\nCriterion 4 — Dataset realism: synthetic / lab-generated traffic")
    print(f"  (This is a known limitation shared by almost all public IDS datasets —")
    print(f"   state it explicitly in your Methodology, as your supervisor requested.)")

    return {
        "dataset": dataset_name,
        "total_rows": total_rows,
        "imbalance_ratio": imbalance_ratio,
        "missing_rate_pct": missing_rate,
        "duplicate_rate_pct": duplicate_rate,
    }


if __name__ == "__main__":
    # Load the CLEANED data (output of script 01)
    cicids_path = os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl")
    if not os.path.exists(cicids_path):
        raise FileNotFoundError(
            "cicids2017_clean.pkl not found. Run 01_load_and_clean_cicids.py first."
        )
    cicids = pd.read_pickle(cicids_path)

    print("Generating class distribution chart...")
    chart_path = os.path.join(RESULTS_DIR, "cicids2017_class_distribution.png")
    plot_class_distribution(
        cicids, "Label", "CICIDS2017 — Class Distribution (after cleaning)", chart_path
    )

    # Run the suitability report for CICIDS2017
    cicids_report = suitability_report(cicids, "Label", "CICIDS2017 (cleaned)")

    # Load the cleaned UNSW-NB15 data (output of script 02)
    unsw_path = os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl")
    if not os.path.exists(unsw_path):
        raise FileNotFoundError(
            "unsw_nb15_clean.pkl not found. Run 02_load_and_clean_unsw.py first."
        )
    unsw = pd.read_pickle(unsw_path)

    # attack_cat is blank for normal rows, so fill those in. Without this, pandas
    # skips blank values when counting classes, and the normal rows (about 95% of
    # the data) would silently vanish from the imbalance ratio. Also strip stray
    # spaces, merge the two spellings of the same attack ("Backdoors" and
    # "Backdoor"), and treat any leftover empty or "nan" text as Normal too.
    unsw["attack_label"] = (
        unsw["attack_cat"]
        .fillna("Normal")
        .astype(str)
        .str.strip()
        .replace({"Backdoors": "Backdoor", "nan": "Normal", "": "Normal"})
    )

    # Run the same suitability checks on UNSW-NB15 and draw its chart
    unsw_report = suitability_report(unsw, "attack_label", "UNSW-NB15 (cleaned)")

    unsw_chart_path = os.path.join(RESULTS_DIR, "unsw_nb15_class_distribution.png")
    plot_class_distribution(
        unsw, "attack_label", "UNSW-NB15 — Class Distribution (after cleaning)", unsw_chart_path
    )

    # Save both reports in one file so exact figures can be quoted later
    report_df = pd.DataFrame([cicids_report, unsw_report])
    report_path = os.path.join(RESULTS_DIR, "dataset_suitability_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nSaved suitability numbers to: {report_path}")
    print("\n--> Copy these numbers into your Methodology section's dataset validation paragraph.")
