"""
14_repeated_runs_and_reverse.py
=================================
WHAT THIS SCRIPT DOES:
    It completes the two parts of your results that are still thin.

    PART A  (78 features, within CICIDS2017)
        Your main models (scripts 05 to 07) were each run only once. This part
        retrains Isolation Forest and the Autoencoder several times, with a
        different random seed each time, and records the mean and the range
        of every metric. Models use all 78 features, exactly as in scripts 05
        and 06, and are tested on the same Tuesday to Friday test set.

    PART B  (5 shared features, both directions)
        Scripts 10, 11 and 13 tested CICIDS2017 -> UNSW-NB15. This part adds:
          C. train on UNSW-NB15, test within UNSW-NB15   (repeated with seeds;
             before, this was a single unseeded run)
          D. train on UNSW-NB15, test on CICIDS2017      (the reverse direction)
        for both models, so you can say whether the cross-dataset failure
        happens in one direction only or in both.

WHY SEEDS:
    A neural network starts from random weights, so two runs of the same code
    give slightly different results. Reporting a mean and a range shows how
    much of a difference is real and how much is just run-to-run noise.

SETTINGS TO KNOW:
    - Isolation Forest uses contamination 0.10. Contamination only moves its
      built-in cutoff, so AUC is the same for 0.05, 0.10 and 0.15.
    - The Autoencoder's threshold is the 95th percentile of the error on
      its own benign TRAINING data, as in script 07.
    - To use more runs, edit SEEDS below, for example [42, 43, 44, 45, 46].

HOW TO RUN THIS:
    python3 src/14_repeated_runs_and_reverse.py b     (Part B only, faster)
    python3 src/14_repeated_runs_and_reverse.py a     (Part A only, slower)
    python3 src/14_repeated_runs_and_reverse.py       (both)

    Needs scripts 01, 02, 04, 06 and 10 in the src folder, and 01, 02 and 04
    already run. Part A trains the big autoencoder once per seed, so it can
    take an hour or more. Keep the laptop plugged in and awake. Results are
    saved after every seed, so nothing is lost if you stop it.
"""

import os
import sys
import time
import gc
import importlib.util

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, RESULTS_DIR, COMMON_FEATURE_MAP,
    AUTOENCODER_EPOCHS, AUTOENCODER_BATCH_SIZE, AUTOENCODER_EARLY_STOP_PATIENCE,
)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

SEEDS = [42, 43, 44]
CONTAMINATION = 0.10
MAX_TRAIN_ROWS = 500_000
MICROSECONDS_PER_SECOND = 1_000_000
METRICS = ["auc_roc", "precision", "recall", "f1_score", "false_positive_rate"]


def load_script(filename, module_name):
    """Our scripts start with numbers, so Python's normal 'import' cannot load
    them. Load them by file path to reuse their functions."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SRC_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script06 = load_script("06_train_autoencoder.py", "script06")        # build_autoencoder
script10 = load_script("10_cross_dataset_diagnostics.py", "script10")  # train_on, reconstruction_error


# ----------------------------------------------------------------------------
# Scoring helpers
# ----------------------------------------------------------------------------
def metrics(name, seed, y_true, scores, flagged):
    """AUC from the continuous scores, plus precision, recall, F1 and false
    positive rate from the yes/no decisions."""
    flagged = np.asarray(flagged).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "experiment": name,
        "seed": seed,
        "n_rows": len(y_true),
        "attack_rate": float(np.mean(y_true)),
        "auc_roc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "f1_score": f1_score(y_true, flagged, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def score_iforest(name, seed, model, X_scaled, y):
    """Isolation Forest: higher score = more anomalous, and a row is flagged
    when it falls below the model's own built-in cutoff (same rule as
    model.predict(X) == -1, without scoring the data twice)."""
    raw = model.score_samples(X_scaled)
    return metrics(name, seed, y, -raw, raw < model.offset_)


def score_autoencoder(name, seed, model, train_error, X_scaled, y):
    """Autoencoder: the anomaly score is the rebuild error, and a row is
    flagged when it is above the 95th percentile of the TRAINING error."""
    scores = script10.reconstruction_error(model, X_scaled)
    return metrics(name, seed, y, scores, scores >= np.percentile(train_error, 95))


def save_results(rows, stem):
    """Save per-run results and a mean / lowest / highest summary."""
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(RESULTS_DIR, f"{stem}_runs.csv"), index=False)
    summary = runs.groupby("experiment")[METRICS].agg(["mean", "min", "max"]).round(3)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(os.path.join(RESULTS_DIR, f"{stem}_summary.csv"))
    return summary


def print_run(row):
    print(f"    {row['experiment']}: AUC={row['auc_roc']:.3f}  recall={row['recall']:.3f}  "
          f"FPR={row['false_positive_rate']:.3f}")


# ----------------------------------------------------------------------------
# PART A: 78 features, within CICIDS2017
# ----------------------------------------------------------------------------
def run_part_a():
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test_binary.npy"))
    print(f"Part A: train {X_train.shape}, test {X_test.shape}, attack rate {y_test.mean():.2%}")

    rows = []
    for seed in SEEDS:
        start = time.time()
        print(f"\n===== Part A, seed {seed} =====")
        keras.utils.set_random_seed(seed)

        iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                              random_state=seed, n_jobs=-1).fit(X_train)
        rows.append(score_iforest("A1: Isolation Forest, 78 features, within CICIDS2017",
                                  seed, iso, X_test, y_test))
        print_run(rows[-1])

        autoencoder = script06.build_autoencoder(X_train.shape[1])
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=AUTOENCODER_EARLY_STOP_PATIENCE,
            restore_best_weights=True)
        autoencoder.fit(X_train, X_train, epochs=AUTOENCODER_EPOCHS,
                        batch_size=AUTOENCODER_BATCH_SIZE, validation_split=0.1,
                        callbacks=[early_stop], verbose=0)
        train_error = script10.reconstruction_error(autoencoder, X_train)
        rows.append(score_autoencoder("A2: Autoencoder, 78 features, within CICIDS2017",
                                      seed, autoencoder, train_error, X_test, y_test))
        print_run(rows[-1])

        save_results(rows, "repeated_runs_78_features")   # saved after every seed
        print(f"    (seed {seed} took {(time.time() - start) / 60:.1f} minutes)")

    summary = save_results(rows, "repeated_runs_78_features")
    print("\nPART A SUMMARY (mean, lowest, highest across seeds):")
    print(summary.T.to_string())


# ----------------------------------------------------------------------------
# PART B: 5 shared features, within UNSW-NB15 and the reverse direction
# ----------------------------------------------------------------------------
def run_part_b():
    cicids = pd.read_pickle(os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl"))
    unsw = pd.read_pickle(os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl"))

    cic_cols = list(COMMON_FEATURE_MAP.keys())
    unsw_cols = list(COMMON_FEATURE_MAP.values())

    # CICIDS2017 test set (Tuesday to Friday), duration converted to seconds
    cic_X = cicids[cic_cols].astype(float).copy()
    cic_X["Flow Duration"] = cic_X["Flow Duration"] / MICROSECONDS_PER_SECOND
    is_monday = cicids["source_file"].str.contains("Monday", case=False).values
    cic_test_X = cic_X.values[~is_monday]
    cic_test_y = (cicids.loc[~is_monday, "Label"] != "BENIGN").astype(int).values
    del cicids, cic_X
    gc.collect()

    # UNSW-NB15 (deduplicated by script 02): shared features, already in seconds
    unsw_X = unsw[unsw_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    unsw_X = unsw_X.fillna(unsw_X.median()).values
    unsw_y = (unsw["label"] != 0).astype(int).values
    del unsw
    gc.collect()
    print(f"Part B: UNSW rows {len(unsw_y):,} (attack rate {unsw_y.mean():.2%}), "
          f"CICIDS test rows {len(cic_test_y):,} (attack rate {cic_test_y.mean():.2%})")

    normal_idx = np.where(unsw_y == 0)[0]
    attack_idx = np.where(unsw_y == 1)[0]

    rows = []
    for seed in SEEDS:
        start = time.time()
        print(f"\n===== Part B, seed {seed} =====")
        keras.utils.set_random_seed(seed)

        # Train on 80% of UNSW normal traffic; hold out the other 20% of normal
        # traffic, plus all attacks, as the within-UNSW test set.
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(normal_idx)
        split = int(0.8 * len(shuffled))
        train_idx = shuffled[:split]
        held_out_normal = shuffled[split:][:MAX_TRAIN_ROWS]
        test_idx = np.concatenate([held_out_normal, attack_idx])

        # Autoencoder (script 10's train_on fits the scaler on training data only)
        scaler, autoencoder, train_error = script10.train_on(unsw_X[train_idx])

        # Isolation Forest on the same training sample, with the same scaler
        sample = script10.sample_rows(unsw_X[train_idx], MAX_TRAIN_ROWS)
        iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION,
                              random_state=seed, n_jobs=-1).fit(scaler.transform(sample))

        within_X = scaler.transform(unsw_X[test_idx])
        within_y = unsw_y[test_idx]
        reverse_X = scaler.transform(cic_test_X)

        rows.append(score_autoencoder("C1: Autoencoder, within UNSW-NB15", seed,
                                      autoencoder, train_error, within_X, within_y))
        print_run(rows[-1])
        rows.append(score_iforest("C2: Isolation Forest, within UNSW-NB15", seed,
                                  iso, within_X, within_y))
        print_run(rows[-1])
        rows.append(score_autoencoder("D1: Autoencoder, UNSW-NB15 -> CICIDS2017", seed,
                                      autoencoder, train_error, reverse_X, cic_test_y))
        print_run(rows[-1])
        rows.append(score_iforest("D2: Isolation Forest, UNSW-NB15 -> CICIDS2017", seed,
                                  iso, reverse_X, cic_test_y))
        print_run(rows[-1])

        save_results(rows, "reverse_direction")
        print(f"    (seed {seed} took {(time.time() - start) / 60:.1f} minutes)")

    summary = save_results(rows, "reverse_direction")
    print("\nPART B SUMMARY (mean, lowest, highest across seeds):")
    print(summary.T.to_string())


if __name__ == "__main__":
    part = sys.argv[1].lower() if len(sys.argv) > 1 else "both"
    if part not in ("a", "b", "both"):
        raise SystemExit("Use: python3 src/14_repeated_runs_and_reverse.py [a|b]")
    if part in ("b", "both"):
        run_part_b()
    if part in ("a", "both"):
        run_part_a()
    print("\nDone. Results are in your results folder (repeated_runs_78_features_* "
          "and reverse_direction_*).")
