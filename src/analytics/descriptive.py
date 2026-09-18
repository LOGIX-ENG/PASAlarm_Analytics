import pandas as pd


def alarms_by_area(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("Area")
        .size()
        .reset_index(name="Alarm Count")
        .sort_values(
            "Alarm Count",
            ascending=False,
        )
    )


def alarms_by_field(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby(
            ["Area", "Field"]
        )
        .size()
        .reset_index(name="Alarm Count")
        .sort_values(
            "Alarm Count",
            ascending=False,
        )
    )


def alarms_by_asset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "Asset" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(
            ["Area", "Field", "Asset"]
        )
        .size()
        .reset_index(name="Alarm Count")
        .sort_values(
            "Alarm Count",
            ascending=False,
        )
    )


def alarms_by_tag(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby(
            ["Area", "Field", "Tag"]
        )
        .size()
        .reset_index(name="Alarm Count")
        .sort_values(
            "Alarm Count",
            ascending=False,
        )
    )


def alarms_by_priority(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("Priority")
        .size()
        .reset_index(name="Alarm Count")
        .sort_values("Priority")
    )


def alarms_by_state(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("Alarm State")
        .size()
        .reset_index(name="Alarm Count")
        .sort_values(
            "Alarm Count",
            ascending=False,
        )
    )


def alarms_by_type(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("Type")
        .size()
        .reset_index(name="Alarm Count")
        .sort_values(
            "Alarm Count",
            ascending=False,
        )
    )


def alarms_by_hour(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("Hour")
        .size()
        .reset_index(name="Alarm Count")
        .sort_values("Hour")
    )


def alarms_by_day(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby("Date")
        .size()
        .reset_index(name="Alarm Count")
        .sort_values("Date")
    )


def alarms_by_day_of_week(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby(
            ["DayOfWeek", "DayName"]
        )
        .size()
        .reset_index(name="Alarm Count")
        .sort_values("DayOfWeek")
    )