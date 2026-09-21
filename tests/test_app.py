import numpy as np
import pandas as pd

from app import (
    FEATURE_COLUMNS,
    create_ml_windows,
    evaluate_clustering,
    extract_features,
    predict_holdout_membership,
    run_hdbscan,
)


def make_test_data(periods=120):
    """
    Create a small synthetic alarm dataset with the same
    columns used by the application.
    """
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=periods,
        freq="min",
    )

    data = pd.DataFrame(
        {
            "Time / Date": timestamps,
            "Tag": [
                f"TAG{i % 10:02d}"
                for i in range(periods)
            ],
            "Priority": [
                300 if i % 3 == 0 else 500
                for i in range(periods)
            ],
            "Type": ["Alarm"] * periods,
            "Quality": ["Good"] * periods,
            "Alarm State": [
                ["Active", "Acked", "Normal"][i % 3]
                for i in range(periods)
            ],
            "Area": [
                "Area A" if i < periods // 2 else "Area B"
                for i in range(periods)
            ],
            "Field": [
                f"FIELD{(i % 4) + 1:04d}"
                for i in range(periods)
            ],
        }
    )

    return data


# ============================================================
# DATA TESTS
# ============================================================

def test_test_data_created():
    """Synthetic test data should contain records."""
    df = make_test_data()

    assert len(df) > 0


def test_required_source_columns_exist():
    """Synthetic data should contain the columns used by the app."""
    df = make_test_data()

    required_columns = {
        "Time / Date",
        "Tag",
        "Priority",
        "Alarm State",
        "Area",
        "Field",
    }

    assert required_columns.issubset(df.columns)


def test_priority_is_numeric():
    """Priority should be numerical."""
    df = make_test_data()

    assert pd.api.types.is_numeric_dtype(df["Priority"])


def test_timestamp_is_datetime():
    """Time / Date should contain datetime values."""
    df = make_test_data()

    assert pd.api.types.is_datetime64_any_dtype(
        df["Time / Date"]
    )


# ============================================================
# MACHINE-LEARNING WINDOW TESTS
# ============================================================

def test_create_ml_windows_returns_data():
    """Alarm events should produce five-minute observations."""
    df = make_test_data()

    windows = create_ml_windows(df)

    assert len(windows) > 0


def test_ml_feature_columns_exist():
    """All required ML features should be created."""
    df = make_test_data()

    windows = create_ml_windows(df)

    for feature in FEATURE_COLUMNS:
        assert feature in windows.columns


def test_alarm_count_matches_source_records():
    """
    The total AlarmCount across all windows should equal
    the number of source alarm records.
    """
    df = make_test_data()

    windows = create_ml_windows(df)

    assert windows["AlarmCount"].sum() == len(df)


def test_alarm_rate_calculation():
    """
    AlarmRatePerMinute should equal AlarmCount divided
    by the fixed five-minute window.
    """
    df = make_test_data()

    windows = create_ml_windows(df)

    expected = windows["AlarmCount"] / 5

    assert np.allclose(
        windows["AlarmRatePerMinute"],
        expected,
    )


def test_alarm_state_counts():
    """
    Active, Acked, and Normal counts should account
    for all records in the synthetic dataset.
    """
    df = make_test_data()

    windows = create_ml_windows(df)

    state_total = (
        windows["ActiveCount"].sum()
        + windows["AckedCount"].sum()
        + windows["NormalCount"].sum()
    )

    assert state_total == len(df)


def test_high_priority_count():
    """
    HighPriorityCount should represent records with
    Priority greater than or equal to 500.
    """
    df = make_test_data()

    windows = create_ml_windows(df)

    expected = (df["Priority"] >= 500).sum()

    assert windows["HighPriorityCount"].sum() == expected


# ============================================================
# FEATURE EXTRACTION TESTS
# ============================================================

def test_extract_features_returns_expected_columns():
    """Feature extraction should return only ML features."""
    df = make_test_data()

    windows = create_ml_windows(df)
    features = extract_features(windows)

    assert list(features.columns) == FEATURE_COLUMNS


def test_features_have_no_missing_values():
    """ML features should not contain missing values."""
    df = make_test_data()

    windows = create_ml_windows(df)
    features = extract_features(windows)

    assert not features.isna().any().any()


def test_features_are_finite():
    """ML features should not contain infinity."""
    df = make_test_data()

    windows = create_ml_windows(df)
    features = extract_features(windows)

    assert np.isfinite(
        features.to_numpy()
    ).all()


# ============================================================
# DESCRIPTIVE HDBSCAN TESTS
# ============================================================

def test_run_hdbscan_returns_cluster_column():
    """Descriptive HDBSCAN should assign cluster labels."""
    df = make_test_data(periods=600)

    windows = create_ml_windows(df)

    results, scaled_features = run_hdbscan(windows)

    assert "Cluster" in results.columns
    assert len(results) == len(windows)
    assert len(scaled_features) == len(windows)


def test_hdbscan_cluster_labels_are_valid():
    """
    HDBSCAN cluster labels should be integers.
    Noise is represented by -1.
    """
    df = make_test_data(periods=600)

    windows = create_ml_windows(df)

    results, _ = run_hdbscan(windows)

    assert pd.api.types.is_integer_dtype(
        results["Cluster"]
    )

    assert results["Cluster"].min() >= -1


# ============================================================
# CLUSTER EVALUATION TESTS
# ============================================================

def test_evaluate_clustering_with_valid_clusters():
    """
    Evaluation should return Silhouette and
    Davies-Bouldin metrics when at least two clusters exist.
    """
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2],
        ]
    )

    labels = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    metrics = evaluate_clustering(
        features,
        labels,
    )

    assert metrics is not None
    assert "silhouette" in metrics
    assert "davies_bouldin" in metrics
    assert metrics["silhouette"] > 0
    assert metrics["davies_bouldin"] >= 0


def test_evaluate_clustering_ignores_noise():
    """
    Noise observations labeled -1 should not prevent
    evaluation of valid clusters.
    """
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.2],
            [50.0, 50.0],
        ]
    )

    labels = np.array(
        [0, 0, 0, 1, 1, 1, -1]
    )

    metrics = evaluate_clustering(
        features,
        labels,
    )

    assert metrics is not None
    assert metrics["n_clusters_scored"] == 2


def test_evaluate_clustering_requires_two_clusters():
    """
    Evaluation should return None when fewer than
    two non-noise clusters exist.
    """
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
        ]
    )

    labels = np.array(
        [0, 0, 0]
    )

    metrics = evaluate_clustering(
        features,
        labels,
    )

    assert metrics is None


# ============================================================
# HOLDOUT / APPROXIMATE PREDICT TESTS
# ============================================================

def test_holdout_prediction_returns_expected_results():
    """
    The chronological holdout workflow should return
    training and holdout information.
    """
    df = make_test_data(periods=1200)

    windows = create_ml_windows(df)

    result = predict_holdout_membership(windows)

    assert "train_windows" in result
    assert "holdout_windows" in result
    assert "holdout_results" in result
    assert "matched" in result
    assert "unmatched" in result
    assert "train_metrics" in result


def test_holdout_uses_all_observations():
    """
    Training and holdout observations together should
    equal the complete machine-learning dataset.
    """
    df = make_test_data(periods=1200)

    windows = create_ml_windows(df)

    result = predict_holdout_membership(windows)

    total = (
        result["train_windows"]
        + result["holdout_windows"]
    )

    assert total == len(windows)


def test_holdout_is_approximately_80_20():
    """
    The chronological split should use approximately
    80 percent for training.
    """
    df = make_test_data(periods=1200)

    windows = create_ml_windows(df)

    result = predict_holdout_membership(windows)

    expected_train = int(
        len(windows) * 0.8
    )

    assert result["train_windows"] == expected_train


def test_holdout_results_have_prediction_columns():
    """
    approximate_predict should produce cluster membership
    and membership-strength information.
    """
    df = make_test_data(periods=1200)

    windows = create_ml_windows(df)

    result = predict_holdout_membership(windows)

    holdout = result["holdout_results"]

    assert "PredictedCluster" in holdout.columns
    assert "MembershipStrength" in holdout.columns
    assert "MatchesKnownPattern" in holdout.columns


def test_holdout_predictions_account_for_all_records():
    """
    Every holdout observation should either match an
    established cluster or be classified as noise.
    """
    df = make_test_data(periods=1200)

    windows = create_ml_windows(df)

    result = predict_holdout_membership(windows)

    classified = (
        result["matched"]
        + result["unmatched"]
    )

    assert classified == result["holdout_windows"]


def test_membership_strength_is_valid():
    """
    HDBSCAN membership strength should remain between
    zero and one.
    """
    df = make_test_data(periods=1200)

    windows = create_ml_windows(df)

    result = predict_holdout_membership(windows)

    strengths = result[
        "holdout_results"
    ]["MembershipStrength"]

    assert strengths.between(0, 1).all()