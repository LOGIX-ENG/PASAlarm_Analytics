import pandas as pd


def create_alarm_windows(
    df: pd.DataFrame,
    window: str = "5min",
) -> pd.DataFrame:
    """
    Aggregate alarm activity into Area/Field time windows.

    Each ML observation represents:
        one Area + one Field + one time window
    """

    data = df.copy()

    data["Window"] = (
        data["Time / Date"].dt.floor(window)
    )

    group_columns = [
        "Window",
        "Area",
        "Field",
    ]

    aggregations = {
        "AlarmCount": ("Tag", "count"),
        "UniqueTags": ("Tag", "nunique"),
        "AveragePriority": ("Priority", "mean"),
        "MaximumPriority": ("Priority", "max"),
        "ActiveCount": ("IsActive", "sum"),
        "AckedCount": ("IsAcked", "sum"),
        "NormalCount": ("IsNormal", "sum"),
    }

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

    high_priority = (
        data.assign(
            HighPriority=data["Priority"] >= 500
        )
        .groupby(group_columns)["HighPriority"]
        .sum()
        .reset_index(name="HighPriorityCount")
    )

    result = result.merge(
        high_priority,
        on=group_columns,
        how="left",
    )

    result["AlarmRatePerMinute"] = (
        result["AlarmCount"] / 5.0
    )

    return result