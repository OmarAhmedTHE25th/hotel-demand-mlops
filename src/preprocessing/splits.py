"""Reproducible train/validation/test splits for project tasks."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.cleaning import regression_filter
from src.features.feature_sets import get_features


def _validate_split_sizes(config: dict[str, Any]) -> tuple[float, float, int]:
    """Validate configured holdout sizes and return sizes and random seed."""
    test_size = float(config.get("test_size", 0.15))
    val_size = float(config.get("val_size", 0.15))
    random_seed = int(config.get("random_seed", 42))
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")
    return val_size, test_size, random_seed


def _make_splits(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    task: str,
    target: str,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split row positions while retaining the source dataframe indices."""
    val_size, test_size, random_seed = _validate_split_sizes(config)
    positions = pd.RangeIndex(len(df)).to_numpy()
    targets = df[target].to_numpy()
    stratify_values = targets if stratify else None
    train_positions, holdout_positions = train_test_split(
        positions,
        test_size=val_size + test_size,
        random_state=random_seed,
        stratify=stratify_values,
    )

    holdout_targets = targets[holdout_positions]
    holdout_stratify = holdout_targets if stratify else None
    relative_test_size = test_size / (val_size + test_size)
    val_positions, test_positions = train_test_split(
        holdout_positions,
        test_size=relative_test_size,
        random_state=random_seed,
        stratify=holdout_stratify,
    )

    features = get_features(task, config)
    required_columns = features + [target]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataframe is missing required columns: {missing_columns}")

    def select(positions_to_select: Any) -> tuple[pd.DataFrame, pd.Series]:
        """Return one split's feature frame and target series."""
        selected = df.iloc[positions_to_select]
        return selected.loc[:, features].copy(), selected.loc[:, target].copy()

    X_train, y_train = select(train_positions)
    X_val, y_val = select(val_positions)
    X_test, y_test = select(test_positions)
    split_positions = {
        "train": train_positions.tolist(),
        "val": val_positions.tolist(),
        "test": test_positions.tolist(),
    }

    splits_dir = Path(config.get("splits_dir", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)
    indices_path = splits_dir / f"{task}_indices.json"
    indices_path.write_text(
        json.dumps(
            {
                "index_type": "zero-based row positions in the input dataframe",
                **split_positions,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"{task.title()} split shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"Split row positions saved to: {indices_path}")

    if task == "classification":
        for split_name, y_split in (
            ("train", y_train),
            ("val", y_val),
            ("test", y_test),
        ):
            print(f"{split_name} IsCanceled distribution:")
            print(y_split.value_counts(normalize=True).sort_index())
    else:
        for split_name, y_split in (
            ("train", y_train),
            ("val", y_val),
            ("test", y_test),
        ):
            print(
                f"{split_name} ADR mean/std: "
                f"{y_split.mean():.4f} / {y_split.std():.4f}"
            )

    return X_train, X_val, X_test, y_train, y_val, y_test


def split_classification(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create reproducible stratified 70/15/15 splits.

    The test set must not be used for tuning or model selection.
    """
    return _make_splits(
        df,
        config,
        task="classification",
        target="IsCanceled",
        stratify=True,
    )


def split_regression(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """ADR-filter then create reproducible random 70/15/15 splits.

    The test set must not be used for tuning or model selection.
    """
    filtered = regression_filter(df, adr_max=float(config.get("adr_max", 500)))
    return _make_splits(
        filtered,
        config,
        task="regression",
        target="ADR",
        stratify=False,
    )
