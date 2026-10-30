"""Configuration and dataset loading helpers."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml


DATASET_URL = (
    "https://raw.githubusercontent.com/afaq-ahmad/"
    "Hotel-booking-demand-Case-Study/master/H2.csv"
)
EXPECTED_SHAPE = (79330, 31)


def load_config(path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Load project settings from a YAML file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {config_path}")
    return config


def load_h2(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load H2.csv and verify that it has the expected dimensions."""
    if path is None:
        configured_path = load_config().get("data_path", "data/H2.csv")
        csv_path = Path(configured_path)
    else:
        csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Download H2.csv from {DATASET_URL}"
        )

    data = pd.read_csv(csv_path)
    assert data.shape == EXPECTED_SHAPE, (
        f"Expected dataset shape {EXPECTED_SHAPE}, but found {data.shape} "
        f"at '{csv_path}'."
    )
    return data
