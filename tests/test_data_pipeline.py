"""Tests for the strict real-observation dataset contract."""

from pathlib import Path

import pytest

from src.data.dataset import RainfallSpatioTemporalDataset, split_chronological_sequences, split_years


def test_real_dataset_requires_observations() -> None:
    with pytest.raises(FileNotFoundError, match="Real INSAT and GPM files"):
        RainfallSpatioTemporalDataset([], [])


def test_calendar_partitions_are_disjoint() -> None:
    assert set(split_years("train")).isdisjoint(split_years("val"))
    assert set(split_years("val")).isdisjoint(split_years("test"))


def test_chronological_split_has_no_overlap() -> None:
    splits = split_chronological_sequences([f"2023-06-{index:02d}" for index in range(1, 21)])
    assert set(splits["train_indices"]).isdisjoint(splits["val_indices"])
    assert set(splits["train_indices"]).isdisjoint(splits["test_indices"])
    assert max(splits["train_indices"]) < min(splits["val_indices"])
