from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr

class RainfallSpatioTemporalDataset(Dataset):
    def __init__(
        self,
        satellite_sequence_paths: List[List[Path]],
        gpm_target_paths: List[Path],
        norm_stats_path: Optional[Path] = None,
        patch_size: int = 128,
        is_train: bool = True
    ):
        if len(satellite_sequence_paths) != len(gpm_target_paths):
            raise ValueError("Satellite sequences and GPM targets count must match.")
        if len(satellite_sequence_paths) == 0:
            raise FileNotFoundError("Real dataset files are required.")

        self.satellite_sequence_paths = satellite_sequence_paths
        self.gpm_target_paths = gpm_target_paths
        self.patch_size = patch_size
        self.is_train = is_train
        self.norm_stats = self._load_or_init_norm_stats(norm_stats_path)

    def _load_or_init_norm_stats(self, path: Optional[Path]) -> Dict[str, float]:
        if path and Path(path).exists():
            with open(path, "r") as f:
                return json.load(f)
        return {
            "mean_BT_TIR1": 265.0, "std_BT_TIR1": 25.0,
            "mean_BT_TIR2": 263.0, "std_BT_TIR2": 24.0,
            "mean_BT_WV": 235.0, "std_BT_WV": 12.0,
            "mean_BT_MIR": 280.0, "std_BT_MIR": 20.0,
            "mean_BTD_SplitWindow": 2.0, "std_BTD_SplitWindow": 3.0,
            "mean_BTD_Overshoot": -30.0, "std_BTD_Overshoot": 15.0
        }

    def __len__(self) -> int:
        return len(self.satellite_sequence_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_paths = self.satellite_sequence_paths[idx]
        target_path = self.gpm_target_paths[idx]

        seq_tensors = []
        for p in seq_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing sequence file: {p}")
            ds = xr.open_dataset(p)
            channels = [
                (ds["BT_TIR1"].values - self.norm_stats["mean_BT_TIR1"]) / self.norm_stats["std_BT_TIR1"],
                (ds["BT_TIR2"].values - self.norm_stats["mean_BT_TIR2"]) / self.norm_stats["std_BT_TIR2"],
                (ds["BT_WV"].values - self.norm_stats["mean_BT_WV"]) / self.norm_stats["std_BT_WV"],
                (ds["BT_MIR"].values - self.norm_stats["mean_BT_MIR"]) / self.norm_stats["std_BT_MIR"],
                (ds["BTD_SplitWindow"].values - self.norm_stats["mean_BTD_SplitWindow"]) / self.norm_stats["std_BTD_SplitWindow"],
                (ds["BTD_Overshoot"].values - self.norm_stats["mean_BTD_Overshoot"]) / self.norm_stats["std_BTD_Overshoot"]
            ]
            seq_tensors.append(np.nan_to_num(np.stack(channels, axis=0), nan=0.0))
            ds.close()

        seq_tensor = np.stack(seq_tensors, axis=0)
        target_ds = xr.open_dataset(target_path)
        cls_mask = np.nan_to_num(target_ds["heavy_rain_mask"].values, nan=0.0)
        qpe_map = np.nan_to_num(target_ds["qpe_rate"].values, nan=0.0)
        target_ds.close()

        _, _, H, W = seq_tensor.shape
        top = 0 if not self.is_train or H <= self.patch_size else np.random.randint(0, H - self.patch_size + 1)
        left = 0 if not self.is_train or W <= self.patch_size else np.random.randint(0, W - self.patch_size + 1)

        patch_x = seq_tensor[:, :, top:top+self.patch_size, left:left+self.patch_size]
        patch_cls = cls_mask[np.newaxis, top:top+self.patch_size, left:left+self.patch_size]
        patch_qpe = qpe_map[np.newaxis, top:top+self.patch_size, left:left+self.patch_size]

        return torch.from_numpy(patch_x).float(), torch.from_numpy(patch_cls).float(), torch.from_numpy(patch_qpe).float()

def split_chronological_sequences(all_timestamps: List[str], train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict[str, List[int]]:
    n = len(all_timestamps)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    return {
        "train_indices": list(range(0, t_end)),
        "val_indices": list(range(t_end, v_end)),
        "test_indices": list(range(v_end, n))
    }
