"""Deterministic data cleaning rules for the City Hotel H2 dataset."""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based cleaning and report row counts and class balance.

    The input dataframe is not modified. Missing Children values count as zero
    only when evaluating whether a booking has any guests.
    """
    cleaned = df.copy()
    rows_before = len(cleaned)

    for column in ("Agent", "Company"):
        is_null_string = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .str.upper()
            .eq("NULL")
            .fillna(False)
        )
        cleaned[column] = cleaned[column].mask(is_null_string, pd.NA)
    cleaned["ReservationStatusDate"] = pd.to_datetime(
        cleaned["ReservationStatusDate"], errors="coerce"
    )
    print("Rows removed by NULL/date normalization: 0")

    rows_before_rule = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    print(f"Rows removed by exact duplicate rule: {rows_before_rule - len(cleaned)}")

    rows_before_rule = len(cleaned)
    guest_count = (
        cleaned["Adults"] + cleaned["Children"].fillna(0) + cleaned["Babies"]
    )
    cleaned = cleaned.loc[guest_count.ne(0)]
    print(f"Rows removed by zero-guests rule: {rows_before_rule - len(cleaned)}")

    rows_before_rule = len(cleaned)
    total_nights = (
        cleaned["StaysInWeekendNights"] + cleaned["StaysInWeekNights"]
    )
    cleaned = cleaned.loc[total_nights.ne(0)]
    print(f"Rows removed by zero-nights rule: {rows_before_rule - len(cleaned)}")

    cleaned = cleaned.reset_index(drop=True)
    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Final shape: {cleaned.shape}")
    print("IsCanceled class balance (normalized):")
    print(cleaned["IsCanceled"].value_counts(normalize=True).sort_index())
    return cleaned


def regression_filter(df: pd.DataFrame, adr_max: float) -> pd.DataFrame:
    """Keep only rows with ADR in the range (0, adr_max] for regression."""
    filtered = df.loc[df["ADR"].gt(0) & df["ADR"].le(adr_max)].copy()
    print(f"Rows removed by regression ADR filter: {len(df) - len(filtered)}")
    return filtered
