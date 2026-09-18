import pandas as pd


def _window_duration_minutes(window: str) -> float:
    """
    Convert a Pandas time-window string into minutes.

    Examples:
        5min  -> 5
        10min -> 10
        15min -> 15
        30min -> 30
        1h    -> 60
    """

    try:
        offset = pd.tseries.frequencies.to_offset(window)
        duration = offset.nanos / (60 * 1_000_000_000)

        if duration <= 0:
            raise ValueError

        return duration

    except (ValueError, TypeError):
        raise ValueError(
            f"Invalid window size: {window}"
        )


def create_alarm_windows(
    df: pd.DataFrame,
    window: str = "5min",
) -> pd.DataFrame:
    """
    Aggregate alarm activity into Area/Field time windows.

    Each ML observation represents:

        one Area
        +
        one Field
        +
        one time window

    Parameters
    ----------
    df : pd.DataFrame
        Prepared alarm-event dataset.

    window : str
        Pandas-compatible time window such as:
        "5min", "10min", "15min", or "30min".

    Returns
    -------
    pd.DataFrame
        Aggregated ML observations.
    """

    required_columns = [
        "Time / Date",
        "Tag",
        "Priority",
        "Area",
        "Field",
        "IsActive",
        "IsAcked",
        "IsNormal",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns required for window creation: "
            f"{missing_columns}"
        )

    window_minutes = _window_duration_minutes(window)

    data = df.copy()

    # ------------------------------------------------------------
    # Ensure timestamps are valid
    # ------------------------------------------------------------

    data["Time / Date"] = pd.to_datetime(
        data["Time / Date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["Time / Date"]
    )

    if data.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------
    # Create time window
    # ------------------------------------------------------------

    data["Window"] = (
        data["Time / Date"]
        .dt.floor(window)
    )

    group_columns = [
        "Window",
        "Area",
        "Field",
    ]

    # ------------------------------------------------------------
    # Base aggregations
    # ------------------------------------------------------------

    aggregations = {
        "AlarmCount": ("Tag", "count"),
        "UniqueTags": ("Tag", "nunique"),
        "AveragePriority": ("Priority", "mean"),
        "MaximumPriority": ("Priority", "max"),
        "ActiveCount": ("IsActive", "sum"),
        "AckedCount": ("IsAcked", "sum"),
        "NormalCount": ("IsNormal", "sum"),
    }

    # Asset is optional because some versions of the
    # anonymized dataset may not contain it.
    if "Asset" in data.columns:
        aggregations["UniqueAssets"] = (
            "Asset",
            "nunique",
        )

    result = (
        data.groupby(group_columns)
        .agg(**aggregations)
        .reset_index()
    )

    # ------------------------------------------------------------
    # High-priority alarms
    #
    # The current project defines priority >= 500 as
    # high priority based on the existing project methodology.
    # ------------------------------------------------------------

    high_priority = (
        data.assign(
            HighPriority=data["Priority"] >= 500
        )
        .groupby(group_columns)["HighPriority"]
        .sum()
        .reset_index(
            name="HighPriorityCount"
        )
    )

    result = result.merge(
        high_priority,
        on=group_columns,
        how="left",
    )

    # ------------------------------------------------------------
    # Dynamic alarm rate
    # ------------------------------------------------------------

    result["AlarmRatePerMinute"] = (
        result["AlarmCount"]
        / window_minutes
    )

    # ------------------------------------------------------------
    # Clean numeric output
    # ------------------------------------------------------------

    numeric_columns = [
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

    if "UniqueAssets" in result.columns:
        numeric_columns.append(
            "UniqueAssets"
        )

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            ).fillna(0)

    return result