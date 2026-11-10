"""
test_pipeline.py
==================
Automated tests for the parts of the pipeline whose correctness the results
depend on. Run them from the project folder with:

    python3 -m pytest tests -v

Each test checks ONE specific behaviour and says in its name what it expects.
If you change a script and a test fails, the test is telling you that you
changed something the results rely on.

Our scripts have names starting with numbers (01_, 02_ ...), which Python's
normal 'import' cannot handle, so the helper below loads them by file path.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)


def load_script(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(SRC, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def s01():
    return load_script("01_load_and_clean_cicids.py", "s01")


@pytest.fixture(scope="module")
def s02():
    return load_script("02_load_and_clean_unsw.py", "s02")


@pytest.fixture(scope="module")
def s07():
    return load_script("07_evaluate_models.py", "s07")


@pytest.fixture(scope="module")
def s08():
    return load_script("08_threshold_sensitivity.py", "s08")


@pytest.fixture(scope="module")
def s10():
    return load_script("10_cross_dataset_diagnostics.py", "s10")


# ---------------------------------------------------------------------------
# Script 01: cleaning CICIDS2017
# ---------------------------------------------------------------------------
def test_cicids_infinite_values_are_replaced_by_the_median(s01):
    df = pd.DataFrame({
        "Flow Bytes/s": [1.0, np.inf, 3.0, 5.0],
        "Label": ["BENIGN", "BENIGN", "DoS", "DoS"],
        "source_file": ["Monday"] * 4,
    })
    cleaned = s01.clean_cicids(df)
    assert not np.isinf(cleaned["Flow Bytes/s"]).any()
    assert cleaned.loc[1, "Flow Bytes/s"] == 3.0   # median of 1, 3, 5


def test_cicids_label_whitespace_is_stripped(s01):
    df = pd.DataFrame({
        "f": [1.0, 2.0],
        "Label": [" BENIGN", "BENIGN "],
        "source_file": ["Monday", "Monday"],
    })
    cleaned = s01.clean_cicids(df)
    assert set(cleaned["Label"]) == {"BENIGN"}


def test_cicids_exact_duplicate_rows_are_removed(s01):
    row = {"f": 1.0, "Label": "BENIGN", "source_file": "Monday"}
    df = pd.DataFrame([row, row, {"f": 2.0, "Label": "BENIGN", "source_file": "Monday"}])
    assert len(s01.clean_cicids(df)) == 2


def test_cicids_identical_rows_from_different_days_are_merged(s01):
    # The duplicate check ignores 'source_file', so an identical flow that
    # appears on two different days is kept only once (the first copy).
    df = pd.DataFrame({
        "f": [1.0, 1.0],
        "Label": ["BENIGN", "BENIGN"],
        "source_file": ["Monday", "Tuesday"],
    })
    cleaned = s01.clean_cicids(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["source_file"] == "Monday"


# ---------------------------------------------------------------------------
# Script 02: loading and cleaning UNSW-NB15
# ---------------------------------------------------------------------------
def test_unsw_column_list_has_49_unique_names_ending_with_label(s02):
    names = s02.UNSW_COLUMN_NAMES
    assert len(names) == 49
    assert len(set(names)) == 49
    assert names[-2:] == ["attack_cat", "label"]


def test_shared_feature_map_points_at_real_unsw_columns(s02):
    from config import COMMON_FEATURE_MAP
    assert set(COMMON_FEATURE_MAP.values()) <= set(s02.UNSW_COLUMN_NAMES)
    assert len(set(COMMON_FEATURE_MAP.values())) == len(COMMON_FEATURE_MAP)


def test_unsw_cleaning_fixes_infinite_values_but_keeps_attack_cat(s02):
    df = pd.DataFrame({
        "dur": [0.1, np.inf, 0.3],
        "label": [0, 1, 1],
        "attack_cat": [np.nan, "Exploits", "Exploits"],
    })
    cleaned = s02.clean_unsw(df)
    assert not np.isinf(cleaned["dur"]).any()
    assert cleaned["attack_cat"].isna().sum() == 1     # blank for normal rows, left alone
    assert set(cleaned["label"]) == {0, 1}              # label never altered


def test_unsw_cleaning_removes_exact_duplicates(s02):
    row = {"dur": 0.5, "label": 1, "attack_cat": "DoS"}
    df = pd.DataFrame([row, row, {"dur": 0.7, "label": 0, "attack_cat": np.nan}])
    assert len(s02.clean_unsw(df)) == 2


# ---------------------------------------------------------------------------
# Script 04 principle: the scaler must be fitted on training data only
# ---------------------------------------------------------------------------
def test_scaler_fitted_on_training_data_does_not_recentre_test_data():
    train = np.random.default_rng(0).normal(0, 1, (1000, 1))
    test = np.random.default_rng(1).normal(5, 1, (1000, 1))   # a very different distribution
    scaler = StandardScaler().fit(train)
    assert abs(scaler.transform(train).mean()) < 0.01
    assert scaler.transform(test).mean() > 4     # still far from 0, so nothing leaked in


# ---------------------------------------------------------------------------
# Script 07: metrics and scores
# ---------------------------------------------------------------------------
def test_metrics_for_perfect_predictions(s07):
    y = np.array([0, 0, 1, 1])
    result = s07.compute_metrics(y, y.copy(), np.array([0.1, 0.2, 0.8, 0.9]), "m", "t")
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["false_positive_rate"] == 0.0
    assert result["auc_roc"] == 1.0


def test_false_positive_rate_is_wrong_alarms_over_all_normal_rows(s07):
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([1, 0, 0, 0, 1, 1])       # 1 of 4 normal rows flagged
    result = s07.compute_metrics(y, pred, np.array([0.9, 0.1, 0.2, 0.1, 0.8, 0.9]), "m", "t")
    assert result["false_positive_rate"] == pytest.approx(0.25)


def test_isolation_forest_scores_are_higher_for_outliers(s07):
    X = np.random.default_rng(0).normal(0, 1, (500, 3))
    model = IsolationForest(random_state=0).fit(X)
    scores = s07.isolation_forest_scores(model, np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 8.0]]))
    assert scores[1] > scores[0]


# ---------------------------------------------------------------------------
# Script 08: threshold sweep
# ---------------------------------------------------------------------------
def test_higher_threshold_never_increases_recall_or_false_alarms(s08):
    rng = np.random.default_rng(0)
    reference = rng.exponential(1, 2000)
    test = np.concatenate([rng.exponential(1, 1500), rng.exponential(4, 500)])
    y = np.array([0] * 1500 + [1] * 500)
    sweep = s08.threshold_sweep(test, y, [80, 90, 99], reference)
    assert sweep["recall"].is_monotonic_decreasing
    assert sweep["false_positive_rate"].is_monotonic_decreasing


def test_threshold_comes_from_the_reference_scores_not_the_test_scores(s08):
    rng = np.random.default_rng(0)
    reference = rng.exponential(1, 2000)
    test = rng.exponential(5, 1000)
    y = rng.integers(0, 2, 1000)
    sweep = s08.threshold_sweep(test, y, [95], reference)
    assert sweep["threshold_value"].iloc[0] == pytest.approx(np.percentile(reference, 95))


# ---------------------------------------------------------------------------
# Script 10: helper functions
# ---------------------------------------------------------------------------
def test_row_sampling_caps_the_size_and_is_repeatable(s10):
    X = np.arange(1000).reshape(500, 2)
    first = s10.sample_rows(X, 100)
    second = s10.sample_rows(X, 100)
    assert first.shape == (100, 2)
    assert np.array_equal(first, second)
    assert s10.sample_rows(X, 1000).shape == X.shape


def test_small_autoencoder_rebuilds_the_same_number_of_features(s10):
    model = s10.build_small_autoencoder(5)
    assert model.predict(np.zeros((3, 5)), verbose=0).shape == (3, 5)
