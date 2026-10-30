"""Tests for the exploratory data analysis workflow."""

from pathlib import Path

import pytest

from src.data.eda import run_eda


@pytest.mark.skipif(not Path("data/H2.csv").is_file(), reason="H2.csv is not present")
def test_run_eda_creates_summary_and_figures(tmp_path: Path) -> None:
    """EDA should write a text summary and all requested plot files."""
    run_eda(tmp_path)

    expected_files = (
        "eda_summary.txt",
        "adr_histogram.png",
        "lead_time_histogram.png",
        "adr_boxplot.png",
        "lead_time_boxplot.png",
        "cancellation_by_arrival_month.png",
        "cancellation_by_deposit_type.png",
        "cancellation_by_market_segment.png",
        "numeric_correlation_heatmap.png",
        "top_15_countries.png",
    )
    for filename in expected_files:
        assert (tmp_path / filename).is_file()
