"""Tests for deterministic data cleaning rules."""

import numpy as np
import pandas as pd

from src.data.cleaning import clean_data, regression_filter


def make_booking(
    lead_time: int,
    *,
    adults: int = 1,
    children: float = 0,
    babies: int = 0,
    weekend_nights: int = 0,
    week_nights: int = 1,
    agent: str = "12",
    company: str = "34",
    adr: float = 100.0,
) -> dict[str, object]:
    """Create one minimal row using the dataset's original column names."""
    return {
        "LeadTime": lead_time,
        "Adults": adults,
        "Children": children,
        "Babies": babies,
        "StaysInWeekendNights": weekend_nights,
        "StaysInWeekNights": week_nights,
        "Agent": agent,
        "Company": company,
        "ADR": adr,
        "ReservationStatusDate": "2017-01-01",
        "IsCanceled": lead_time % 2,
    }


def sample_bookings() -> pd.DataFrame:
    """Build rows covering duplicate, zero-guest, and zero-night rules."""
    duplicate = make_booking(1, agent=" NULL ", company="NULL")
    return pd.DataFrame(
        [
            duplicate,
            duplicate.copy(),
            make_booking(2, adults=0, children=np.nan, babies=0),
            make_booking(3, weekend_nights=0, week_nights=0),
            make_booking(4, children=np.nan, agent="NULL", company=" NULL "),
        ]
    )


def test_clean_data_removes_exact_duplicates() -> None:
    """Exact duplicate rows are reduced to one occurrence."""
    cleaned = clean_data(sample_bookings())

    assert cleaned["LeadTime"].tolist().count(1) == 1


def test_clean_data_removes_rows_with_zero_guests() -> None:
    """Children NaN counts as zero when evaluating guest totals."""
    cleaned = clean_data(sample_bookings())

    assert 2 not in cleaned["LeadTime"].tolist()


def test_clean_data_removes_rows_with_zero_nights() -> None:
    """Bookings with no weekend or weekday nights are removed."""
    cleaned = clean_data(sample_bookings())

    assert 3 not in cleaned["LeadTime"].tolist()


def test_clean_data_converts_null_strings_without_filling_children() -> None:
    """NULL values become missing, while Children NaN remains unfilled."""
    cleaned = clean_data(sample_bookings())
    row = cleaned.loc[cleaned["LeadTime"].eq(4)].iloc[0]

    assert pd.isna(row["Agent"])
    assert pd.isna(row["Company"])
    assert pd.isna(row["Children"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["ReservationStatusDate"])


def test_regression_filter_keeps_only_positive_adr_up_to_max() -> None:
    """ADR values at or below zero and above the threshold are removed."""
    data = pd.DataFrame(
        {"ADR": [-1.0, 0.0, 0.01, 500.0, 500.01]}
    )

    filtered = regression_filter(data, adr_max=500.0)

    assert filtered["ADR"].tolist() == [0.01, 500.0]
