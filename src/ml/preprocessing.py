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
    """
    Prepare numerical features for HDBSCAN.

    Returns:
        scaled_features
        scaler
        available_features
        feature_data
    """

    available_features = [
        feature
        for feature in ML_FEATURES
        if feature in window_df.columns
    ]

    if not available_features:
        raise ValueError(
            "No ML features are available."
        )

    feature_data = (
        window_df[available_features]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    if len(feature_data) < 2:
        raise ValueError(
            "At least two observations are required "
            "for ML processing."
        )

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        feature_data
    )

    return (
        scaled_features,
        scaler,
        available_features,
        feature_data,
    )