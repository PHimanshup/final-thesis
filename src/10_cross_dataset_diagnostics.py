"""
10_cross_dataset_diagnostics.py
=================================
WHY THIS SCRIPT EXISTS:
    Script 09 gave a cross-dataset AUC below 0.5 (worse than guessing).
    Before that can be reported as "the effect of dataset bias", we have
    to rule out other explanations. The big one: script 09's model uses
    only the handful of features the two datasets share, while the
    within-dataset results from script 07 used all 78 features. So the
    drop from script 07 to script 09 mixes TWO changes together:
        (1) fewer features, and
        (2) a different dataset.
    This script separates them by running three experiments that all use
    the SAME shared features and the SAME small autoencoder:

        A. Train on CICIDS2017 Monday  -> test on CICIDS2017 Tue-Fri
           (same dataset, shared features only)
        B. Train on CICIDS2017 Monday  -> test on UNSW-NB15
           (the cross-dataset test, with the duration unit fixed here)
        C. Train on UNSW-NB15 normal   -> test on UNSW-NB15 held-out
           (same dataset, shared features only, other direction)

    How to read the results:
        - If A is already poor, the shared features carry little signal,
          and the cross-dataset drop is mostly a feature problem.
        - If A is good but B is poor, the drop is due to the dataset change.
        - If C is poor too, the shared features cannot separate UNSW-NB15
          attacks from normal traffic at all, even within UNSW-NB15.

    It also prints the median value of each shared feature for each group
    of traffic, so you can SEE whether the datasets differ in scale or
    units (the most common cause of a meaningless cross-dataset result).

HOW TO RUN THIS:
    python3 src/10_cross_dataset_diagnostics.py
    (Run scripts 01 and 02 first. This takes roughly 10-20 minutes.)
"""

import os
import sys

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE, COMMON_FEATURE_MAP,
    AUTOENCODER_EPOCHS, AUTOENCODER_BATCH_SIZE, AUTOENCODER_EARLY_STOP_PATIENCE,
)

# CICIDS2017 stores 'Flow Duration' in MICROSECONDS, UNSW-NB15 stores 'dur'
# in SECONDS. We convert CICIDS2017 to seconds so both are in the same unit.
MICROSECONDS_PER_SECOND = 1_000_000

# To keep training time reasonable we train on at most this many rows
MAX_TRAIN_ROWS = 500_000


def build_small_autoencoder(input_dim):
    """Same small autoencoder as script 09, so results are comparable."""
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inputs)
    bottleneck = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(bottleneck)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_error(model, X):
    """Mean squared difference between the input and the rebuilt input."""
    rebuilt = model.predict(X, verbose=0, batch_size=4096)
    return np.mean(np.square(X - rebuilt), axis=1)


def sample_rows(X, max_rows):
    """Randomly keep at most max_rows rows (fixed seed so it is repeatable)."""
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(RANDOM_STATE)
    return X[rng.choice(len(X), size=max_rows, replace=False)]


def train_on(X_train):
    """Scale using ONLY this training data, train an autoencoder on it,
    and return (scaler, model, training reconstruction error)."""
    X_train = sample_rows(X_train, MAX_TRAIN_ROWS)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = build_small_autoencoder(X_scaled.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=AUTOENCODER_EARLY_STOP_PATIENCE,
        restore_best_weights=True,
    )
    model.fit(
        X_scaled, X_scaled, epochs=AUTOENCODER_EPOCHS,
        batch_size=AUTOENCODER_BATCH_SIZE, validation_split=0.1,
        callbacks=[early_stop], verbose=0,
    )
    train_error = reconstruction_error(model, X_scaled)
    return scaler, model, train_error


def evaluate(name, trained_on, tested_on, scaler, model, train_error, X_test, y_test):
    """Score the test data and report AUC, plus recall and false positive
    rate at the 95th-percentile threshold taken from TRAINING error."""
    scores = reconstruction_error(model, scaler.transform(X_test))
    threshold = np.percentile(train_error, 95)
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    row = {
        "experiment": name,
        "trained_on": trained_on,
        "tested_on": tested_on,
        "n_test_rows": len(y_test),
        "attack_rate": float(np.mean(y_test)),
        "auc_roc": roc_auc_score(y_test, scores),
        "recall_at_95pct": tp / (tp + fn) if (tp + fn) else 0.0,
        "false_positive_rate_at_95pct": fp / (fp + tn) if (fp + tn) else 0.0,
    }
    print(f"\n{name}: AUC={row['auc_roc']:.3f}  recall={row['recall_at_95pct']:.3f}  "
          f"FPR={row['false_positive_rate_at_95pct']:.3f}  attack rate={row['attack_rate']:.2%}")
    return row


if __name__ == "__main__":
    cicids = pd.read_pickle(os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl"))
    unsw = pd.read_pickle(os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl"))

    cic_cols = list(COMMON_FEATURE_MAP.keys())
    unsw_cols = list(COMMON_FEATURE_MAP.values())
    print(f"Shared features used ({len(cic_cols)}): {cic_cols}")

    # --- CICIDS2017: shared features, duration converted to seconds ----------
    cic_X = cicids[cic_cols].astype(float).copy()
    cic_X["Flow Duration"] = cic_X["Flow Duration"] / MICROSECONDS_PER_SECOND
    cic_X = cic_X.values

    is_monday = cicids["source_file"].str.contains("Monday", case=False).values
    cic_train_X = cic_X[is_monday]
    cic_test_X = cic_X[~is_monday]
    cic_test_y = (cicids.loc[~is_monday, "Label"] != "BENIGN").astype(int).values

    # --- UNSW-NB15: shared features, already in seconds ----------------------
    unsw_X = unsw[unsw_cols].astype(float).values
    unsw_y = (unsw["label"] != 0).astype(int).values
    print(f"\nUNSW-NB15 rows: {len(unsw_y):,}, attack rate: {unsw_y.mean():.2%}")
    print(f"CICIDS2017 test rows: {len(cic_test_y):,}, attack rate: {cic_test_y.mean():.2%}")

    results = []

    # --- Experiments A and B: model trained on CICIDS2017 Monday -------------
    print("\nTraining on CICIDS2017 Monday (shared features only)...")
    scaler_c, model_c, err_c = train_on(cic_train_X)
    results.append(evaluate("A: within CICIDS2017 (shared features)", "CICIDS2017",
                            "CICIDS2017", scaler_c, model_c, err_c, cic_test_X, cic_test_y))
    results.append(evaluate("B: cross-dataset (shared features)", "CICIDS2017",
                            "UNSW-NB15", scaler_c, model_c, err_c, unsw_X, unsw_y))

    # --- Experiment C: model trained on UNSW-NB15 normal traffic -------------
    print("\nTraining on UNSW-NB15 normal traffic (shared features only)...")
    rng = np.random.default_rng(RANDOM_STATE)
    normal_idx = np.where(unsw_y == 0)[0]
    rng.shuffle(normal_idx)
    split = int(0.8 * len(normal_idx))
    train_idx = normal_idx[:split]
    held_out_normal = normal_idx[split:]
    held_out_normal = held_out_normal[:MAX_TRAIN_ROWS]   # cap test size
    attack_idx = np.where(unsw_y == 1)[0]
    test_idx = np.concatenate([held_out_normal, attack_idx])

    scaler_u, model_u, err_u = train_on(unsw_X[train_idx])
    results.append(evaluate("C: within UNSW-NB15 (shared features)", "UNSW-NB15",
                            "UNSW-NB15", scaler_u, model_u, err_u,
                            unsw_X[test_idx], unsw_y[test_idx]))

    results_path = os.path.join(RESULTS_DIR, "cross_dataset_diagnostics.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nSaved experiment results to: {results_path}")

    # --- Feature medians: do the datasets even live on the same scale? --------
    groups = {
        "CICIDS2017 Monday (benign)": cic_train_X,
        "CICIDS2017 test attacks": cic_test_X[cic_test_y == 1],
        "UNSW-NB15 normal": unsw_X[unsw_y == 0],
        "UNSW-NB15 attacks": unsw_X[unsw_y == 1],
    }
    medians = pd.DataFrame(
        {name: np.median(X, axis=0) for name, X in groups.items()},
        index=cic_cols,
    )
    print("\nMedian of each shared feature, by group of traffic")
    print("(Flow Duration is in seconds for all groups):")
    print(medians.to_string(float_format=lambda v: f"{v:,.4f}"))

    medians_path = os.path.join(RESULTS_DIR, "feature_medians_comparison.csv")
    medians.to_csv(medians_path)
    print(f"\nSaved feature medians to: {medians_path}")
