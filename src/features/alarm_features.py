import pandas as pd


def add_alarm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple indicators from recorded alarm states."""

    result = df.copy()

    state = (
        result["Alarm State"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    result["IsActive"] = (
        state.eq("active").astype(int)
    )

    result["IsAcked"] = (
        state.eq("acked").astype(int)
    )

    result["IsNormal"] = (
        state.eq("normal").astype(int)
    )

    return result