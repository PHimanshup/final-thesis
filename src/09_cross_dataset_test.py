"""
09_cross_dataset_test.py
==========================
WHAT THIS SCRIPT DOES:
    This is the heart of your research question. It takes a model trained
    ONLY on CICIDS2017 and tests it, WITHOUT any retraining, on UNSW-NB15
    — a completely different dataset the model has never seen. The
    resulting performance drop (or lack of one) is your main finding.

**** READ THIS BEFORE YOU RUN IT ****
    CICIDS2017 has 78 features. UNSW-NB15 has 49 features, with different
    names. A model trained on 78 CICIDS2017 features literally cannot
    accept a 49-feature UNSW-NB15 row as input — the shapes don't match.

    The fix used here: train a SEPARATE, smaller model using only a
    handful of features that exist in BOTH datasets under different
    names (e.g. CICIDS2017's "Flow Duration" and UNSW-NB15's "dur" both
    measure flow duration). This mapping is defined in config.py as
    COMMON_FEATURE_MAP.

    THIS MAPPING IS A STARTING POINT, NOT A VERIFIED FACT. Run the
    "print both column lists" step below FIRST, check every single
    mapped pair by eye, and fix config.py if anything looks wrong before
    trusting any result this script produces. Document your final,
    checked mapping — and your reasoning for each pair — in your
    Methodology section.

HOW TO RUN THIS:
    python src/09_cross_dataset_test.py

    (Run scripts 01 and 02 first — this script needs their output.)
"""

import os
import sys

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, RESULTS_DIR, RANDOM_STATE, COMMON_FEATURE_MAP,
    AUTOENCODER_EPOCHS, AUTOENCODER_BATCH_SIZE, AUTOENCODER_EARLY_STOP_PATIENCE,
)


def build_small_autoencoder(input_dim):
    """
    A smaller autoencoder than script 06's, because we're now working
    with only a handful of common features instead of the full 78.
    Same idea: compress, then try to rebuild; high rebuild error = anomaly.
    """
    inputs = keras.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inputs)
    bottleneck = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(bottleneck)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_error(model, X):
    reconstructed = model.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


if __name__ == "__main__":
    # --- Load both cleaned datasets ------------------------------------------
    cicids_path = os.path.join(PROCESSED_DIR, "cicids2017_clean.pkl")
    unsw_path = os.path.join(PROCESSED_DIR, "unsw_nb15_clean.pkl")

    if not os.path.exists(cicids_path) or not os.path.exists(unsw_path):
        raise FileNotFoundError(
            "Cleaned data not found. Run 01_load_and_clean_cicids.py and "
            "02_load_and_clean_unsw.py first."
        )

    cicids = pd.read_pickle(cicids_path)
    unsw = pd.read_pickle(unsw_path)

    # --- STEP 1: print both column lists so you can verify the mapping -------
    print("CICIDS2017 columns:")
    print(list(cicids.columns))
    print("\nUNSW-NB15 columns:")
    print(list(unsw.columns))
    print(f"\nCurrent COMMON_FEATURE_MAP (from config.py): {COMMON_FEATURE_MAP}")
    print("\n>>> Stop here and check this mapping against the column lists above")
    print(">>> before trusting anything printed after this point. <<<\n")

    cicids_common_cols = list(COMMON_FEATURE_MAP.keys())
    unsw_common_cols = list(COMMON_FEATURE_MAP.values())

    missing_from_cicids = [c for c in cicids_common_cols if c not in cicids.columns]
    missing_from_unsw = [c for c in unsw_common_cols if c not in unsw.columns]

    if missing_from_cicids or missing_from_unsw:
        print("ERROR: some mapped columns don't exist in the data you loaded.")
        print(f"  Missing from CICIDS2017: {missing_from_cicids}")
        print(f"  Missing from UNSW-NB15:  {missing_from_unsw}")
        print("Fix COMMON_FEATURE_MAP in config.py, then re-run this script.")
        sys.exit(1)

    # --- STEP 2: build the benign-only training set from CICIDS2017 ----------
    # (Only the common features, only Monday's benign rows)
    is_monday = cicids["source_file"].str.contains("Monday", case=False)
    train_benign = cicids[is_monday]

    X_common_train = train_benign[cicids_common_cols].values
    # UNIT FIX: CICIDS2017 'Flow Duration' is microseconds; UNSW-NB15 'dur' is seconds.
    duration_index = cicids_common_cols.index("Flow Duration")
    X_common_train = X_common_train.astype(float)
    X_common_train[:, duration_index] = X_common_train[:, duration_index] / 1_000_000

    # --- STEP 3: scale using only this common-feature training data ----------
    common_scaler = StandardScaler()
    X_common_train_scaled = common_scaler.fit_transform(X_common_train)

    # --- STEP 4: train a small autoencoder on the common features ------------
    print("Training common-feature autoencoder on CICIDS2017 benign data...")
    model = build_small_autoencoder(input_dim=X_common_train_scaled.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=AUTOENCODER_EARLY_STOP_PATIENCE, restore_best_weights=True
    )
    model.fit(
        X_common_train_scaled, X_common_train_scaled,
        epochs=AUTOENCODER_EPOCHS, batch_size=AUTOENCODER_BATCH_SIZE,
        validation_split=0.1, callbacks=[early_stop], verbose=1,
    )

    # Set the threshold using the training data's own error distribution,
    # exactly as in script 07 — this keeps the cross-dataset test
    # methodologically consistent with the within-dataset test
    train_error = reconstruction_error(model, X_common_train_scaled)
    threshold = np.percentile(train_error, 95)

    # --- STEP 5: prepare the UNSW-NB15 labels ---------------------------------
    # UNSW-NB15's label column is usually already 0 = normal, 1 = attack,
    # but we handle the case where it might be text instead (e.g. "Normal")
    label_col = "label" if "label" in unsw.columns else "Label"
    if unsw[label_col].dtype == object:
        y_unsw = (unsw[label_col].astype(str).str.lower() != "normal").astype(int).values
    else:
        y_unsw = (unsw[label_col] != 0).astype(int).values

    print(f"\nUNSW-NB15 attack rate: {y_unsw.mean():.2%}")

    # --- STEP 6: scale UNSW-NB15's common features using the SAME scaler -----
    # fitted on CICIDS2017 training data — never fit a new scaler on the
    # test data, for the same data-leakage reason explained in script 04
    X_common_unsw = unsw[unsw_common_cols].values
    X_common_unsw_scaled = common_scaler.transform(X_common_unsw)

    # --- STEP 7: score UNSW-NB15 with the CICIDS2017-trained model -----------
    unsw_scores = reconstruction_error(model, X_common_unsw_scaled)
    unsw_predictions = (unsw_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_unsw, unsw_predictions).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    cross_dataset_result = {
        "model": "Autoencoder (common features, trained on CICIDS2017)",
        "tested_on": "UNSW-NB15",
        "precision": precision_score(y_unsw, unsw_predictions, zero_division=0),
        "recall": recall_score(y_unsw, unsw_predictions, zero_division=0),
        "f1_score": f1_score(y_unsw, unsw_predictions, zero_division=0),
        "auc_roc": roc_auc_score(y_unsw, unsw_scores),
        "false_positive_rate": false_positive_rate,
    }

    print("\nCross-dataset generalisation result:")
    for key, value in cross_dataset_result.items():
        print(f"  {key}: {value}")

    result_path = os.path.join(RESULTS_DIR, "cross_dataset_result.csv")
    pd.DataFrame([cross_dataset_result]).to_csv(result_path, index=False)
    print(f"\nSaved cross-dataset result to: {result_path}")

    print("\n--> Compare these numbers to the within-dataset results from script 07.")
    print("--> The SIZE OF THE DROP (or lack of one) between the two is your main finding —")
    print("    write this comparison up directly in your Testing and Conclusions sections.")
