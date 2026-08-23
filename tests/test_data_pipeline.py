"""Focused tests for the synthetic rainfall data pipeline."""

from datetime import datetime, timedelta

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.data.dataset import RainfallSpatioTemporalDataset, split_years


def test_dummy_batch_shapes_and_finite_values() -> None:
    dataset = RainfallSpatioTemporalDataset(
        split="train", dummy=True, patch_size=(128, 128), dummy_channels=2
    )
    inputs, targets = dataset[0]
    batch_inputs = torch.stack([inputs, dataset[1][0]])
    batch_mask = torch.stack([targets["mask"], dataset[1][1]["mask"]])

    assert tuple(batch_inputs.shape) == (2, 4, 2, 128, 128)
    assert tuple(batch_mask.shape) == (2, 1, 128, 128)
    assert torch.isfinite(batch_inputs).all()
    assert torch.isfinite(batch_mask).all()
    assert torch.isfinite(targets["qpe"]).all()


def test_strict_split_boundaries_have_no_temporal_leakage() -> None:
    dataset = RainfallSpatioTemporalDataset(split="val", dummy=True)
    allowed_years = set(split_years("val"))
    for target_timestamp in dataset.sample_timestamps:
        assert target_timestamp.year in allowed_years
    for target_index in dataset.indices:
        window = dataset.timestamps[target_index - 3 : target_index + 2]
        assert {timestamp.year for timestamp in window} == allowed_years


def test_split_rejects_windows_crossing_calendar_boundary() -> None:
    timestamps = [
        datetime(2022, 12, 31, 23, 0) + timedelta(minutes=30 * index)
        for index in range(5)
    ]
    features = np.zeros((5, 1, 128, 128), dtype="float32")
    rainfall = np.zeros((5, 128, 128), dtype="float32")
    with pytest.raises(ValueError, match="No valid samples"):
        RainfallSpatioTemporalDataset(
            features, rainfall, timestamps, split="val", patch_size=(128, 128)
        )
