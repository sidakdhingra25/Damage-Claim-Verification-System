"""Pure CSV lookup for user history risk flags."""

from __future__ import annotations

import pandas as pd


def get_history_risk_flags(user_id: str, history_df: pd.DataFrame) -> list[str]:
    matches = history_df.loc[history_df["user_id"] == user_id]
    if matches.empty:
        return []

    raw_flags = str(matches.iloc[0]["history_flags"])
    return [
        flag
        for flag in (part.strip() for part in raw_flags.split(";"))
        if flag and flag.lower() != "none"
    ]
