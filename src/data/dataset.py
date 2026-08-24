from pathlib import Path
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import glob


def split_years(split: str) -> set[int]:
    partitions = {
        "train": {2019, 2020, 2021},
        "val": {2022},
        "test": {2023},
    }
    try:
        return partitions[split]
    except KeyError as exc:
        raise ValueError("split must be 'train', 'val', or 'test'") from exc


def split_chronological_sequences(all_timestamps: List[str], train_ratio: float = 0.70, val_ratio: float = 0.15) -> Dict[str, List[int]]:
    if not 0.0 <= train_ratio <= 1.0 or not 0.0 <= val_ratio <= 1.0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio and val_ratio must be non-negative and sum to at most 1")
    n = len(all_timestamps)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return {
        "train_indices": list(range(0, train_end)),
        "val_indices": list(range(train_end, val_end)),
        "test_indices": list(range(val_end, n)),
    }

def discover_real_samples(data_dir: str = "data/raw"):
    insat_files = sorted(glob.glob(f"{data_dir}/insat3d/*.nc") + glob.glob(f"{data_dir}/insat3d/*.h5"))
    gpm_files = sorted(glob.glob(f"{data_dir}/gpm_imerg/*.nc") + glob.glob(f"{data_dir}/gpm_imerg/*.HDF5") + glob.glob(f"{data_dir}/gpm_imerg/*.nc4"))

    if len(insat_files) < 4 or len(gpm_files) < 1:
        raise FileNotFoundError(f"Need at least 4 INSAT files and 1 GPM file under {data_dir}. Found: INSAT={len(insat_files)}, GPM={len(gpm_files)}")

    seq_groups = []
    for i in range(len(gpm_files)):
        group = [Path(p) for p in insat_files[i*4 : (i+1)*4]]
        if len(group) == 4:
            seq_groups.append(group)
            
    target_paths = [Path(p) for p in gpm_files[:len(seq_groups)]]
    return seq_groups, target_paths

class RainfallSpatioTemporalDataset(Dataset):
    def __init__(self, satellite_sequence_paths: List[List[Path]], gpm_target_paths: List[Path], patch_size: int = 128, is_train: bool = True):
        if not satellite_sequence_paths or not gpm_target_paths:
            raise FileNotFoundError("Real INSAT and GPM files are required")
        self.satellite_sequence_paths = satellite_sequence_paths
        self.gpm_target_paths = gpm_target_paths
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self):
        return len(self.satellite_sequence_paths)

    def __getitem__(self, idx: int):
        seq_paths = self.satellite_sequence_paths[idx]
        target_path = self.gpm_target_paths[idx]

        seq_tensors = []
        for p in seq_paths:
            ds = xr.open_dataset(p)
            
            # Dynamic Channel Extraction
            def get_var(names, default_val):
                for n in names:
                    if n in ds.data_vars or n in ds:
                        return ds[n].values
                return np.full((128, 128), default_val, dtype=np.float32)

            t1 = get_var(["BT_TIR1", "tir1", "t2m", "air", "rain"], 265.0)
            t2 = get_var(["BT_TIR2", "tir2"], 263.0)
            wv = get_var(["BT_WV", "wv"], 235.0)
            mir = get_var(["BT_MIR", "mir"], 280.0)

            channels = [
                (t1 - 265.0) / 25.0,
                (t2 - 263.0) / 24.0,
                (wv - 235.0) / 12.0,
                (mir - 280.0) / 20.0,
                ((t1 - t2) - 2.0) / 3.0,
                ((wv - t1) + 30.0) / 15.0
            ]
            frame_arr = np.nan_to_num(np.stack(channels, axis=0), nan=0.0).astype(np.float32)
            seq_tensors.append(frame_arr)
            ds.close()

        seq_tensor = np.stack(seq_tensors, axis=0) # [4, 6, H, W]

        # Target dynamic variable reading
        target_ds = xr.open_dataset(target_path)
        rain_var_candidates = ["qpe_rate", "rain", "rf", "precipitationCal", "precip", "precipitation"]
        target_val = None
        for cand in rain_var_candidates:
            if cand in target_ds.data_vars or cand in target_ds:
                target_val = target_ds[cand].values
                break
        
        if target_val is None:
            # Fallback to first available data variable
            first_var = list(target_ds.data_vars.keys())[0]
            target_val = target_ds[first_var].values

        qpe_map = np.nan_to_num(target_val, nan=0.0).astype(np.float32)
        if "heavy_rain_mask" in target_ds.data_vars:
            cls_mask = np.nan_to_num(target_ds["heavy_rain_mask"].values, nan=0.0).astype(np.float32)
        else:
            cls_mask = (qpe_map >= 7.5).astype(np.float32)
            
        target_ds.close()

        # Handle dimension shapes & spatial padding
        if qpe_map.ndim == 3:
            qpe_map = qpe_map[0]
            cls_mask = cls_mask[0]

        _, _, H, W = seq_tensor.shape
        pad_h = max(0, self.patch_size - H)
        pad_w = max(0, self.patch_size - W)
        if pad_h > 0 or pad_w > 0:
            seq_tensor = np.pad(seq_tensor, ((0,0), (0,0), (0, pad_h), (0, pad_w)), mode='edge')
            cls_mask = np.pad(cls_mask, ((0, pad_h), (0, pad_w)), mode='edge')
            qpe_map = np.pad(qpe_map, ((0, pad_h), (0, pad_w)), mode='edge')

        patch_x = seq_tensor[:, :, :self.patch_size, :self.patch_size]
        patch_cls = cls_mask[np.newaxis, :self.patch_size, :self.patch_size]
        patch_qpe = qpe_map[np.newaxis, :self.patch_size, :self.patch_size]

        return torch.from_numpy(patch_x).float(), torch.from_numpy(patch_cls).float(), torch.from_numpy(patch_qpe).float()
