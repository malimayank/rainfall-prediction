"""Real-file and deterministic synthetic rainfall datasets."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_SPLIT_YEARS = {"train": range(2018, 2023), "val": range(2023, 2024), "test": range(2024, 2025)}


def split_years(split: str) -> range:
    if split not in DEFAULT_SPLIT_YEARS:
        raise ValueError(f"split must be one of {tuple(DEFAULT_SPLIT_YEARS)}")
    return DEFAULT_SPLIT_YEARS[split]


def split_chronological_sequences(all_timestamps: list[str], train_ratio: float = 0.70, val_ratio: float = 0.15) -> dict[str, list[int]]:
    """Create non-overlapping chronological index partitions."""
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio > 1:
        raise ValueError("invalid train/validation ratios")
    train_end = int(len(all_timestamps) * train_ratio)
    val_end = int(len(all_timestamps) * (train_ratio + val_ratio))
    return {"train_indices": list(range(train_end)), "val_indices": list(range(train_end, val_end)), "test_indices": list(range(val_end, len(all_timestamps)))}


def _dummy_arrays(channels: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
    timestamps = [datetime(year, 6, 1) + timedelta(minutes=30 * index) for year in range(2018, 2025) for index in range(32)]
    generator = np.random.default_rng(42)
    features = generator.normal(240.0, 15.0, (len(timestamps), channels, height, width)).astype("float32")
    rainfall = generator.gamma(1.5, 4.0, (len(timestamps), height, width)).astype("float32")
    rainfall += np.maximum(features[:, 0] - 250.0, 0.0) * 0.15
    return features, rainfall, timestamps


class RainfallSpatioTemporalDataset(Dataset):
    """Dataset supporting real xarray files and synthetic sliding windows."""

    def __init__(self, satellite_sequence_paths: Optional[List[List[Path]]] = None, gpm_target_paths: Optional[List[Path]] = None,
                 norm_stats_path: Optional[Path] = None, patch_size: int | tuple[int, int] = 128, is_train: bool = True,
                 features: Any | None = None, rainfall: Any | None = None, timestamps: list[datetime | str] | None = None,
                 split: str = "train", sequence_length: int = 4, lead_time_steps: int = 1, lead_time_minutes: int = 60,
                 heavy_rain_threshold: float = 7.5, dummy: bool = False, dummy_channels: int = 3) -> None:
        if isinstance(satellite_sequence_paths, np.ndarray) and isinstance(gpm_target_paths, np.ndarray):
            features, rainfall, timestamps = satellite_sequence_paths, gpm_target_paths, norm_stats_path
            satellite_sequence_paths = None
            gpm_target_paths = None
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.heavy_rain_threshold = heavy_rain_threshold
        self.sequence_length = sequence_length
        self.lead_time_steps = lead_time_steps
        self.lead_time_minutes = lead_time_minutes
        self._real = satellite_sequence_paths is not None or gpm_target_paths is not None
        if self._real:
            if not satellite_sequence_paths or gpm_target_paths is None or len(satellite_sequence_paths) != len(gpm_target_paths):
                raise ValueError("Satellite sequences and GPM targets count must match and cannot be empty.")
            self.satellite_sequence_paths = satellite_sequence_paths
            self.gpm_target_paths = gpm_target_paths
            self.is_train = is_train
            self.norm_stats = self._load_stats(norm_stats_path)
            return
        if dummy:
            features, rainfall, timestamps = _dummy_arrays(dummy_channels, self.patch_size[0], self.patch_size[1])
        if features is None or rainfall is None or timestamps is None:
            raise ValueError("features, rainfall, timestamps, or dummy=True are required")
        self.features = np.nan_to_num(np.asarray(features, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
        self.rainfall = np.nan_to_num(np.asarray(rainfall, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
        self.timestamps = [datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value for value in timestamps]
        self.split = split
        allowed = set(split_years(split))
        self.indices = []
        for target_index in range(sequence_length - 1, len(self.timestamps) - lead_time_steps):
            inputs = range(target_index - sequence_length + 1, target_index + 1)
            if all(self.timestamps[index].year in allowed for index in inputs) and self.timestamps[target_index + lead_time_steps].year in allowed:
                self.indices.append(target_index)
        if not self.indices:
            raise ValueError(f"No valid samples found for split {split!r}")
        self.sample_timestamps = [self.timestamps[index + lead_time_steps] for index in self.indices]
        values = self.features[self.indices]
        self.mean = np.nanmean(values, axis=(0, 2, 3), keepdims=True).astype("float32")
        self.std = np.nanstd(values, axis=(0, 2, 3), keepdims=True).astype("float32")
        self.std = np.where(self.std > 1e-6, self.std, 1.0)

    def _load_stats(self, path: Optional[Path]) -> dict[str, float]:
        if path and Path(path).exists():
            return json.loads(Path(path).read_text())
        return {"mean_BT_TIR1": 265.0, "std_BT_TIR1": 25.0, "mean_BT_TIR2": 263.0, "std_BT_TIR2": 24.0,
                "mean_BT_WV": 235.0, "std_BT_WV": 12.0, "mean_BT_MIR": 280.0, "std_BT_MIR": 20.0,
                "mean_BTD_SplitWindow": 2.0, "std_BTD_SplitWindow": 3.0, "mean_BTD_Overshoot": -30.0, "std_BTD_Overshoot": 15.0}

    def __len__(self) -> int:
        return len(self.satellite_sequence_paths) if self._real else len(self.indices)

    def __getitem__(self, index: int):
        if self._real:
            import xarray as xr
            channels = []
            for path in self.satellite_sequence_paths[index]:
                with xr.open_dataset(path) as dataset:
                    names = [("BT_TIR1", "mean_BT_TIR1", "std_BT_TIR1"), ("BT_TIR2", "mean_BT_TIR2", "std_BT_TIR2"),
                             ("BT_WV", "mean_BT_WV", "std_BT_WV"), ("BT_MIR", "mean_BT_MIR", "std_BT_MIR"),
                             ("BTD_SplitWindow", "mean_BTD_SplitWindow", "std_BTD_SplitWindow"), ("BTD_Overshoot", "mean_BTD_Overshoot", "std_BTD_Overshoot")]
                    channels.append([np.nan_to_num((dataset[name].values - self.norm_stats[mean]) / self.norm_stats[std]) for name, mean, std in names])
            with xr.open_dataset(self.gpm_target_paths[index]) as target:
                mask = np.nan_to_num(target["heavy_rain_mask"].values, nan=0.0)
                qpe = np.nan_to_num(target["qpe_rate"].values, nan=0.0)
            return torch.tensor(np.asarray(channels), dtype=torch.float32), torch.tensor(mask[None], dtype=torch.float32), torch.tensor(qpe[None], dtype=torch.float32)
        target_index = self.indices[index]
        input_indices = range(target_index - self.sequence_length + 1, target_index + 1)
        inputs = (self.features[list(input_indices)] - self.mean) / self.std
        target = self.rainfall[target_index + self.lead_time_steps]
        height, width = self.patch_size
        inputs = inputs[..., :height, :width]
        target = target[:height, :width]
        return torch.from_numpy(np.nan_to_num(inputs).copy()), {"mask": torch.from_numpy((target >= self.heavy_rain_threshold).astype("float32")[None]), "qpe": torch.from_numpy(target[None].copy())}


RainfallSpatioTemporalDataModule = None
