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


def validate_alarm_data(df: pd.DataFrame) -> None:
    """Validate that the alarm dataset contains required columns."""

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def load_alarm_data() -> pd.DataFrame:
    """Load and validate the cleaned alarm dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Alarm data file was not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    validate_alarm_data(df)

    return df