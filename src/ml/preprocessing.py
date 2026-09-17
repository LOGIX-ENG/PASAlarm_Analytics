import pandas as pd

from sklearn.preprocessing import StandardScaler


ML_FEATURES = [
    "AlarmCount",
    "UniqueTags",
    "AveragePriority",
    "MaximumPriority",
    "ActiveCount",
    "AckedCount",
    "NormalCount",
    "HighPriorityCount",
    "AlarmRatePerMinute",
]


def prepare_ml_data(
    window_df: pd.DataFrame,
):
    """Prepare and standardize numerical features for HDBSCAN."""

    available_features = [
        feature
        for feature in ML_FEATURES
        if feature in window_df.columns
    ]

    features = (
        window_df[available_features]
        .copy()
        .fillna(0)
    )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    return (
        scaled_features,
        scaler,
        available_features,
    )