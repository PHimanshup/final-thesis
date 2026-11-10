# Investigating the Impact of Dataset Bias on Unsupervised Network Intrusion Detection Systems

COM7002 Project — Himanshu Pant

## What this project does

Trains two unsupervised anomaly detection models (Isolation Forest and an
Autoencoder) on benign-only CICIDS2017 traffic, evaluates them on held-out
mixed CICIDS2017 traffic, then tests whether that performance survives being
applied, without retraining, to a second, independently built dataset
(UNSW-NB15).

## One-time setup

1. **Install Python 3.10 or newer**, if you don't already have it.
2. **Open this folder in VS Code**: File → Open Folder → select
   `network-ids-bias-project`.
3. **Open a terminal in VS Code**: Terminal → New Terminal.
4. **Create a virtual environment** (keeps this project's packages separate
   from everything else on your machine):
   ```
   python -m venv venv
   ```
5. **Activate it**:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

   You'll know it worked because your terminal prompt will show `(venv)`
   at the start of the line.
6. **Install the required packages**:
   ```
   pip install -r requirements.txt
   ```
7. **Add your data**:
   - Put the CICIDS2017 day-by-day CSV files into `data/raw/cicids2017/`
   - Put the UNSW-NB15 CSV file(s) into `data/raw/unsw_nb15/`

   (These folders are empty on purpose — the actual datasets are too large
   to include here and aren't ours to redistribute. Download them from the
   original sources named in your proposal.)

## Running the pipeline

Run the scripts **in order**, from the project's root folder (not from
inside `src/`). Each one depends on the previous one's output:

```
python src/01_load_and_clean_cicids.py
python src/02_load_and_clean_unsw.py
python src/03_eda_and_suitability.py
python src/04_split_and_scale.py
python src/05_train_isolation_forest.py
python src/06_train_autoencoder.py
python src/07_evaluate_models.py
python src/08_threshold_sensitivity.py
python src/09_cross_dataset_test.py
```

Each script prints what it's doing as it runs, and tells you where it saved
its output. You don't need to run them all in one sitting — intermediate
results are saved to `data/processed/` and `models/`, so you can close VS
Code and pick up later from wherever you left off, without re-running
earlier scripts.

## What goes in your report

- **`results/dataset_suitability_report.csv`** and the printed output of
  script 03 → your Methodology's dataset validation paragraph
- **`results/model_evaluation_results.csv`** and **`results/roc_curves.png`**
  → your Testing section
- **`results/threshold_sensitivity_table.csv`** and
  **`results/threshold_sensitivity_plot.png`** → your Testing section,
  threshold sensitivity sub-section
- **`results/cross_dataset_result.csv`** → your Testing and Conclusions
  sections — this is your main finding
- **`results/autoencoder_training_history.png`** → your Development section,
  as evidence the model converged properly

## Before you trust the cross-dataset result (script 09)

Read the big comment block at the top of `src/09_cross_dataset_test.py`
first. CICIDS2017 and UNSW-NB15 don't share a feature space, so script 09
uses a manually defined mapping (`COMMON_FEATURE_MAP` in `config.py`)
between features that measure similar things under different names. This
mapping is a starting point, not a verified fact — check it against your
actual column names before trusting any result downstream of it.

## Folder structure

```
network-ids-bias-project/
├── config.py                  <- all file paths and settings, edit here
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                   <- put your downloaded CSVs here
│   └── processed/             <- cleaned data, saved automatically
├── models/                    <- trained models, saved automatically
├── results/                   <- plots and CSVs for your report, saved automatically
└── src/
    ├── 01_load_and_clean_cicids.py
    ├── 02_load_and_clean_unsw.py
    ├── 03_eda_and_suitability.py
    ├── 04_split_and_scale.py
    ├── 05_train_isolation_forest.py
    ├── 06_train_autoencoder.py
    ├── 07_evaluate_models.py
    ├── 08_threshold_sensitivity.py
    └── 09_cross_dataset_test.py
```
