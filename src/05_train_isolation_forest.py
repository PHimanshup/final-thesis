"""
05_train_isolation_forest.py
==============================
WHAT THIS SCRIPT DOES:
    Trains three Isolation Forest models — one for each "contamination"
    value stated in your proposal (0.05, 0.10, 0.15) — using only the
    scaled, benign training data. Saves all three trained models to disk.

WHAT IS ISOLATION FOREST, IN PLAIN TERMS:
    It builds many random "decision trees" that repeatedly split the data
    on random features at random values. Anomalies (rare, unusual points)
    tend to get isolated into their own tiny branch after just a few
    splits, because there's nothing else like them nearby. Normal points
    need many more splits to isolate, because they're surrounded by
    similar points. The algorithm turns "how many splits it took to
    isolate this point" into an anomaly score.

WHAT "CONTAMINATION" MEANS HERE:
    It's a setting that tells the model roughly what fraction of points
    to treat as anomalies when making its internal normal/anomaly cutoff.
    Since we only train on benign data, this isn't a literal measurement
    of attack proportion — it's closer to a sensitivity dial. That's why
    your proposal tests three different values rather than picking one
    arbitrarily.

HOW TO RUN THIS:
    python src/05_train_isolation_forest.py

    (Run 04_split_and_scale.py first — this script needs its output.)
"""

import os
import sys

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, ISOLATION_FOREST_CONTAMINATIONS


if __name__ == "__main__":
    # Load the scaled training data produced by script 04
    X_train_path = os.path.join(PROCESSED_DIR, "X_train_scaled.npy")
    if not os.path.exists(X_train_path):
        raise FileNotFoundError(
            "X_train_scaled.npy not found. Run 04_split_and_scale.py first."
        )
    X_train = np.load(X_train_path)
    print(f"Loaded training data: {X_train.shape}")

    # Train one Isolation Forest model per contamination value
    for contamination in ISOLATION_FOREST_CONTAMINATIONS:
        print(f"\nTraining Isolation Forest (contamination={contamination})...")

        model = IsolationForest(
            n_estimators=200,       # number of random trees in the forest — more trees
                                     # gives more stable scores, 200 is a common default
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,               # use all available CPU cores to speed up training
        )
        model.fit(X_train)

        # Save each model with a filename that includes its contamination
        # value, so script 07 (evaluation) can load all three separately
        model_filename = f"isolation_forest_c{contamination}.joblib"
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"  Saved: {model_path}")

    print("\nAll three Isolation Forest models trained and saved.")
