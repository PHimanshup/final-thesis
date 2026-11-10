"""
config.py
=========
This file holds every file path and every "tunable" setting used across the
whole project. Every other script imports from here instead of repeating
paths. If you move your data, or want to change a setting, you only ever
need to edit it in ONE place: this file.
"""

import os

# -----------------------------------------------------------------------
# FOLDER PATHS
# -----------------------------------------------------------------------
# BASE_DIR = the folder this config.py file lives in (the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Put your raw CICIDS2017 day-by-day CSV files in this folder
RAW_CICIDS_DIR = os.path.join(BASE_DIR, "data", "raw", "cicids2017")

# Put your raw UNSW-NB15 CSV file(s) in this folder
RAW_UNSW_DIR = os.path.join(BASE_DIR, "data", "raw", "unsw_nb15")

# Cleaned data gets saved here automatically by the scripts (you don't need
# to create anything here yourself)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Plots, metric tables, and anything you'll actually put in your report
# get saved here
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Trained models (Isolation Forest, Autoencoder, the fitted scaler) are
# saved here so you don't have to retrain every time you run a later script
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create any of the above folders that don't exist yet, automatically,
# every time any script runs. This means you never get a confusing
# "folder not found" error.
for folder in [PROCESSED_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------
# Fixing this number means every run of the code gives the same random
# train/test split and the same model initialisation, so your results are
# reproducible (important if your supervisor asks you to re-run something)
RANDOM_STATE = 42

# Three contamination values for Isolation Forest, as stated in your proposal.
# "Contamination" is Isolation Forest's estimate of what fraction of the
# training data is anomalous — since we train on benign-only data, this is
# really a sensitivity setting, not a literal anomaly estimate.
ISOLATION_FOREST_CONTAMINATIONS = [0.05, 0.10, 0.15]

# Autoencoder training settings
AUTOENCODER_EPOCHS = 50
AUTOENCODER_BATCH_SIZE = 256
AUTOENCODER_EARLY_STOP_PATIENCE = 5  # stop training if val_loss doesn't improve for 5 epochs

# A starting-point mapping between CICIDS2017 and UNSW-NB15 columns that
# measure roughly the same thing, used only in script 09 (cross-dataset test).
# CHECK THIS against your actual column names before trusting it — see the
# comments in 09_cross_dataset_test.py for why this matters.
COMMON_FEATURE_MAP = {
    "Flow Duration": "dur",
    "Total Fwd Packets": "spkts",
    "Total Backward Packets": "dpkts",
    "Total Length of Fwd Packets": "sbytes",
    "Total Length of Bwd Packets": "dbytes",
}
