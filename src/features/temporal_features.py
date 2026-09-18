import pandas as pd


def add_temporal_features(
    df: pd.DataFrame,
    timestamp_column: str = "Time / Date",
) -> pd.DataFrame:
    """
    Add temporal features used for descriptive analytics.

    Invalid timestamps are retained as NaT so that data-quality
    reporting can identify them rather than silently hiding them.
    """

    result = df.copy()

    result[timestamp_column] = pd.to_datetime(
        result[timestamp_column],
        errors="coerce",
    )

    result["Year"] = result[timestamp_column].dt.year
    result["Month"] = result[timestamp_column].dt.month
    result["MonthName"] = result[
        timestamp_column
    ].dt.month_name()

    result["Day"] = result[
        timestamp_column
    ].dt.day

    result["DayOfWeek"] = result[
        timestamp_column
    ].dt.dayofweek

    result["DayName"] = result[
        timestamp_column
    ].dt.day_name()

    result["Hour"] = result[
        timestamp_column
    ].dt.hour

    result["Minute"] = result[
        timestamp_column
    ].dt.minute

    result["Date"] = result[
        timestamp_column
    ].dt.date

    result["IsWeekend"] = (
        result["DayOfWeek"] >= 5
    )

    return result