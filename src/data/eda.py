"""Exploratory data analysis for the City Hotel H2 dataset."""

import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.loader import load_h2


MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def _save_figure(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    """Apply a tight layout, save a figure, and release its resources."""
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def run_eda(out_dir: Union[str, Path] = "outputs") -> None:
    """Print a data quality summary and save the requested EDA figures."""
    output_path = Path(out_dir)
    os.makedirs(output_path, exist_ok=True)

    df = load_h2()
    df["ReservationStatusDate"] = pd.to_datetime(
        df["ReservationStatusDate"], errors="coerce"
    )
    numeric_columns = df.select_dtypes(include="number").columns

    summary = io.StringIO()
    with redirect_stdout(summary):
        row_count = len(df)
        print("Missing values (NaN and 'NULL'):")
        for column in ("Agent", "Company", "Country", "Children"):
            null_strings = (
                df[column]
                .astype("string")
                .str.strip()
                .str.upper()
                .eq("NULL")
                .fillna(False)
            )
            missing = df[column].isna() | null_strings
            count = int(missing.sum())
            percentage = count / row_count * 100 if row_count else 0.0
            print(f"  {column}: {count} ({percentage:.2f}%)")

        print(f"\nDuplicate rows: {int(df.duplicated().sum())}")
        print("\nIsCanceled class balance (normalized):")
        print(df["IsCanceled"].value_counts(normalize=True).sort_index())

        categorical_columns = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns
        print("\nCategorical column cardinalities:")
        for column in categorical_columns:
            print(f"  {column}: {df[column].nunique(dropna=True)}")

        total_nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
        total_guests = df["Adults"] + df["Children"] + df["Babies"]
        print("\nUnusual values:")
        print(f"  Adults == 0: {int(df['Adults'].eq(0).sum())}")
        print(f"  ADR <= 0: {int(df['ADR'].le(0).sum())}")
        print(f"  ADR max: {df['ADR'].max()}")
        print(f"  Zero total nights: {int(total_nights.eq(0).sum())}")
        print(f"  Adults + Children + Babies == 0: {int(total_guests.eq(0).sum())}")
        print(f"  Children NaN: {int(df['Children'].isna().sum())}")

        print("\nNumeric column description:")
        print(df[numeric_columns].describe())

    for column, filename, title in (
        ("ADR", "adr_histogram.png", "ADR distribution"),
        ("LeadTime", "lead_time_histogram.png", "LeadTime distribution"),
    ):
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=40)
        ax.set(title=title, xlabel=column, ylabel="Count")
        _save_figure(fig, output_path, filename)

    for column, filename in (
        ("ADR", "adr_boxplot.png"),
        ("LeadTime", "lead_time_boxplot.png"),
    ):
        fig, ax = plt.subplots()
        ax.boxplot(df[column].dropna(), showfliers=True)
        ax.set(title=f"{column} boxplot (including outliers)", ylabel=column)
        _save_figure(fig, output_path, filename)

    df["ArrivalDateMonth"] = pd.Categorical(
        df["ArrivalDateMonth"], categories=MONTHS, ordered=True
    )
    cancellation_groups = (
        (
            "ArrivalDateMonth",
            "cancellation_by_arrival_month.png",
            "Cancellation rate by arrival month",
        ),
        ("DepositType", "cancellation_by_deposit_type.png", "Cancellation rate by deposit type"),
        (
            "MarketSegment",
            "cancellation_by_market_segment.png",
            "Cancellation rate by market segment",
        ),
    )
    for column, filename, title in cancellation_groups:
        rates = df.groupby(column, observed=False)["IsCanceled"].mean().dropna()
        fig, ax = plt.subplots()
        rates.plot(kind="bar", ax=ax)
        ax.set(title=title, xlabel=column, ylabel="Cancellation rate", ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=45)
        _save_figure(fig, output_path, filename)

    fig, ax = plt.subplots(figsize=(12, 10))
    correlation = df[numeric_columns].corr()
    sns.heatmap(correlation, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap of numeric columns")
    _save_figure(fig, output_path, "numeric_correlation_heatmap.png")

    top_countries = df["Country"].value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    top_countries.plot(kind="barh", ax=ax)
    ax.set(title="Top 15 countries by booking count", xlabel="Booking count", ylabel="Country")
    _save_figure(fig, output_path, "top_15_countries.png")

    printed_summary = summary.getvalue()
    print(printed_summary, end="")
    (output_path / "eda_summary.txt").write_text(printed_summary, encoding="utf-8")
