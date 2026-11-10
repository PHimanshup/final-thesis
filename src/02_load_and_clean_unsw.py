"""
02_load_and_clean_unsw.py
===========================
WHAT THIS SCRIPT DOES:
    Same job as 01_load_and_clean_cicids.py, but for the UNSW-NB15 dataset
    instead. We keep this as a SEPARATE script, rather than reusing the
    same code, because the two datasets have different column names and a
    different label format, so trying to force one function to handle both
    would make the code harder to read, not easier.

WHY THIS IS A SEPARATE SCRIPT FROM SCRIPT 01:
    CICIDS2017 and UNSW-NB15 only get combined later (script 09, the
    cross-dataset test). Keeping their loading/cleaning separate here
    mirrors that: you train and test within CICIDS2017 first, completely
    independently of UNSW-NB15, exactly as your Methodology describes.

HOW TO RUN THIS:
    python src/02_load_and_clean_unsw.py
"""

import os
import sys
import glob

import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_UNSW_DIR, PROCESSED_DIR


# IMPORTANT: the raw UNSW-NB15_1.csv ... UNSW-NB15_4.csv files do NOT contain
# a header row — they start directly with data. If we let pandas guess the
# column names, it would silently treat the first row of data as headers,
# which would corrupt every result downstream with no error message.
#
# This is the OFFICIAL column order, taken from the UNSW-NB15_features.csv
# documentation that accompanies the dataset. If your files came from a
# different source and the numbers look wrong after loading (e.g. 'dur'
# contains huge integers instead of small decimal numbers), open one of
# your CSVs in a text editor, compare the first row's values against this
# list in order, and fix any mismatch here before trusting anything else.
UNSW_COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]


def load_all_unsw_files(folder):
    """
    Find every CSV in `folder`, load it WITHOUT treating the first row as
    headers (header=None), assign the official column names ourselves,
    then stack all the files together into one table.
    """
    csv_paths = sorted(glob.glob(os.path.join(folder, "*.csv")))

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {folder}\n"
            "Did you copy your UNSW-NB15 file(s) into data/raw/unsw_nb15/ ?"
        )

    all_parts = []
    for path in csv_paths:
        part = pd.read_csv(
            path,
            header=None,              # <-- critical: there is no header row in the raw file
            names=UNSW_COLUMN_NAMES,  # <-- we supply the real column names ourselves
            low_memory=False,
        )
        print(f"  Loaded {os.path.basename(path):30s} -> {part.shape[0]:,} rows")
        all_parts.append(part)

    combined = pd.concat(all_parts, ignore_index=True)

    # Sanity check: 'dur' (flow duration in seconds) should be a small
    # decimal number, not something like an IP address or a huge integer.
    # If this check fails, the column order above doesn't match your files.
    sample_dur_values = combined["dur"].head(5).tolist()
    print(f"\n  Sanity check — first 5 'dur' values (should be small decimals): {sample_dur_values}")

    return combined


def clean_unsw(df):
    """
    Same cleaning approach as CICIDS2017: fix infinities, fill missing
    numeric values with the median, remove duplicates. UNSW-NB15's
    'label' column is already numeric (0 = normal, 1 = attack).

    Note: 'attack_cat' is blank/NaN for every normal (label=0) row by
    design in the original dataset — that is NOT missing data to fix,
    it's expected, so we deliberately leave it untouched here rather
    than filling it with anything.
    """
    df = df.copy()
    rows_before = len(df)

    # Only touch genuinely numeric feature columns — NOT 'attack_cat',
    # which is text, and NOT 'srcip'/'dstip', which look like numbers
    # but are categorical IP addresses, not something a median makes
    # sense for.
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    numeric_columns = [c for c in numeric_columns if c not in ["label"]]

    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    df = df.drop_duplicates()

    rows_after = len(df)
    print(f"\nCleaning summary:")
    print(f"  Rows before cleaning: {rows_before:,}")
    print(f"  Rows after cleaning:  {rows_after:,}")
    print(f"  Rows removed:         {rows_before - rows_after:,}")

    return df


if __name__ == "__main__":
    print("Loading UNSW-NB15 files...")
    raw_data = load_all_unsw_files(RAW_UNSW_DIR)
    print(f"\nTotal combined shape before cleaning: {raw_data.shape}")

    print("\nCleaning data...")
    clean_data = clean_unsw(raw_data)

    output_path = os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl")
    clean_data.to_pickle(output_path)
    print(f"\nSaved cleaned UNSW-NB15 data to: {output_path}")

    print(f"\nLabel distribution after cleaning ('label'): 0 = normal, 1 = attack")
    print(clean_data["label"].value_counts())
