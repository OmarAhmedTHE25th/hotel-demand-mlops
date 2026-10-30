"""Tests for the H2 dataset loader."""

from pathlib import Path

import pytest

from src.data.loader import EXPECTED_SHAPE, load_h2


@pytest.mark.skipif(not Path("data/H2.csv").is_file(), reason="H2.csv is not present")
def test_load_h2_returns_expected_shape() -> None:
    """The loader should return the H2 dataset with its expected dimensions."""
    data = load_h2()
    assert data.shape == EXPECTED_SHAPE
