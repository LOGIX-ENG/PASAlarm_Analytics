from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "alarms.csv"


REQUIRED_COLUMNS = [
    "Time / Date",
    "Tag",
    "Priority",
    "Type",
    "Quality",
    "Alarm State",
    "Area",
    "Field",
]

EXPECTED_ALARM_STATES = {
    "Active",
    "Acked",
    "Normal",
}


def validate_alarm_data(df: pd.DataFrame) -> None:
    """
    Validate the structure and basic data quality of the alarm dataset.

    Raises:
        ValueError: If required columns are missing or the dataset is empty.
    """

    if df.empty:
        raise ValueError("The alarm dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def prepare_alarm_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and normalize fields used by the application.

    This function does not silently remove records because the
    application should retain visibility into data-quality issues.
    """

    result = df.copy()

    # ------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------

    result["Time / Date"] = pd.to_datetime(
        result["Time / Date"],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Priority
    # ------------------------------------------------------------

    result["Priority"] = pd.to_numeric(
        result["Priority"],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Standardize string fields
    # ------------------------------------------------------------

    string_columns = [
        "Tag",
        "Type",
        "Quality",
        "Alarm State",
        "Area",
        "Field",
    ]

    if "Asset" in result.columns:
        string_columns.append("Asset")

    for column in string_columns:
        result[column] = result[column].astype("string").str.strip()

    return result


def create_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Create a data-quality summary for the application.

    Returns:
        dict containing record counts, missing values,
        invalid values, date range, and other quality indicators.
    """

    total_records = len(df)

    # Timestamp quality
    timestamp_series = pd.to_datetime(
        df["Time / Date"],
        errors="coerce",
    )

    invalid_timestamps = int(timestamp_series.isna().sum())

    valid_timestamps = total_records - invalid_timestamps

    if valid_timestamps > 0:
        min_date = timestamp_series.min()
        max_date = timestamp_series.max()
    else:
        min_date = None
        max_date = None

    # Priority quality
    priority_numeric = pd.to_numeric(
        df["Priority"],
        errors="coerce",
    )

    invalid_priorities = int(priority_numeric.isna().sum())

    # Required field missing values
    missing_values = {}

    for column in REQUIRED_COLUMNS:
        missing_values[column] = int(
            df[column].isna().sum()
            + (
                df[column]
                .astype("string")
                .str.strip()
                .eq("")
                .sum()
                if df[column].dtype.name in ("object", "string")
                else 0
            )
        )

    # Alarm-state quality
    alarm_states = (
        df["Alarm State"]
        .astype("string")
        .str.strip()
    )

    observed_states = set(
        alarm_states.dropna().unique()
    )

    unexpected_states = sorted(
        observed_states - EXPECTED_ALARM_STATES
    )

    missing_alarm_states = int(
        alarm_states.isna().sum()
    )

    # Optional Asset field
    if "Asset" in df.columns:
        missing_assets = int(
            df["Asset"].isna().sum()
            + (
                df["Asset"]
                .astype("string")
                .str.strip()
                .eq("")
                .sum()
            )
        )
    else:
        missing_assets = None

    return {
        "total_records": total_records,
        "valid_timestamps": valid_timestamps,
        "invalid_timestamps": invalid_timestamps,
        "invalid_priorities": invalid_priorities,
        "missing_values": missing_values,
        "missing_alarm_states": missing_alarm_states,
        "unexpected_alarm_states": unexpected_states,
        "observed_alarm_states": sorted(observed_states),
        "missing_assets": missing_assets,
        "date_range_start": min_date,
        "date_range_end": max_date,
        "unique_areas": (
            df["Area"].nunique(dropna=True)
        ),
        "unique_fields": (
            df["Field"].nunique(dropna=True)
        ),
        "unique_assets": (
            df["Asset"].nunique(dropna=True)
            if "Asset" in df.columns
            else 0
        ),
        "unique_tags": (
            df["Tag"].nunique(dropna=True)
        ),
    }


def load_alarm_data() -> pd.DataFrame:
    """
    Load, validate, and prepare the alarm dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Alarm data file was not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    validate_alarm_data(df)

    df = prepare_alarm_data(df)

    return df