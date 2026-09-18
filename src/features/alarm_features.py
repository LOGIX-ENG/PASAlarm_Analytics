import pandas as pd


def add_alarm_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add binary features representing the recorded SCADA alarm state.

    Important:
        These fields describe recorded alarm states.
        They do not infer operator behavior or a complete
        alarm lifecycle.
    """

    result = df.copy()

    state = (
        result["Alarm State"]
        .astype("string")
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