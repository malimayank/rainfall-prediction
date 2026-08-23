import os
from pathlib import Path

files = {
    "src/__init__.py": "",
    "src/utils/__init__.py": "",
    "src/data/__init__.py": "",
    "src/training/__init__.py": "",
    "src/models/__init__.py": "",
    "src/xai/__init__.py": "",
    "src/evaluation/__init__.py": "",
    "tests/__init__.py": "",

    "conftest.py": """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
""",

    "src/utils/geo_utils.py": """from typing import Tuple
import numpy as np
import xarray as xr

def generate_monsoon_target_grid(
    min_lon: float = 65.0,
    min_lat: float = 5.0,
    max_lon: float = 95.0,
    max_lat: float = 38.0,
    resolution: float = 0.04
) -> Tuple[np.ndarray, np.ndarray]:
    lons = np.arange(min_lon, max_lon + resolution / 2.0, resolution)
    lats = np.arange(min_lat, max_lat + resolution / 2.0, resolution)
    return lons, lats

def regrid_to_reference(
    source_da: xr.DataArray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    method: str = "bilinear"
) -> xr.DataArray:
    lat_dim = [d for d in source_da.dims if d.lower() in ["lat", "latitude"]][0]
    lon_dim = [d for d in source_da.dims if d.lower() in ["lon", "longitude"]][0]

    reindexed = source_da.rename({lat_dim: "latitude", lon_dim: "longitude"})
    
    if reindexed.latitude[0] > reindexed.latitude[-1]:
        reindexed = reindexed.reindex(latitude=reindexed.latitude[::-1])
    if reindexed.longitude[0] > reindexed.longitude[-1]:
        reindexed = reindexed.reindex(longitude=reindexed.longitude[::-1])

    return reindexed.interp(latitude=target_lats, longitude=target_lons, method=method)
""",

    "src/data/validator.py": """from pathlib import Path
from typing import Dict, Any, List
import h5py
import numpy as np

class DatasetValidator:
    def __init__(self, min_lon: float = 65.0, min_lat: float = 5.0, max_lon: float = 95.0, max_lat: float = 38.0, min_valid_data_pct: float = 85.0):
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = min_lon, min_lat, max_lon, max_lat
        self.min_valid_data_pct = min_valid_data_pct

    def validate_insat_file(self, filepath: Path) -> Dict[str, Any]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"INSAT-3D file not found: {filepath}")

        report: Dict[str, Any] = {"file": str(filepath.name), "type": "INSAT-3D/3DR HDF5", "status": "VALID", "channels": {}, "errors": []}
        try:
            with h5py.File(filepath, "r") as f:
                required_channels = ["IMG_TIR1", "IMG_TIR2", "IMG_WV", "IMG_MIR"]
                present = list(f.keys())
                for ch in required_channels:
                    matched = [k for k in present if ch in k]
                    if not matched:
                        report["errors"].append(f"Missing channel: {ch}")
                        continue
                    ds_name = matched[0]
                    data = f[ds_name][:]
                    valid_pct = (np.count_nonzero(~np.isnan(data) & (data > 0)) / data.size) * 100.0
                    report["channels"][ch] = {"shape": list(data.shape), "valid_pct": round(valid_pct, 2)}
                    if valid_pct < self.min_valid_data_pct:
                        report["errors"].append(f"{ch} valid pixel ratio ({valid_pct:.1f}%) < {self.min_valid_data_pct}%")
        except Exception as e:
            report["status"] = "INVALID"
            report["errors"].append(str(e))

        if report["errors"]:
            report["status"] = "INVALID"
        return report

    def validate_gpm_file(self, filepath: Path) -> Dict[str, Any]:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"GPM file not found: {filepath}")

        report: Dict[str, Any] = {"file": str(filepath.name), "type": "GPM IMERG", "status": "VALID", "errors": []}
        try:
            with h5py.File(filepath, "r") as f:
                if "Grid" not in f or "precipitationCal" not in f["Grid"]:
                    report["errors"].append("Missing 'Grid/precipitationCal' in GPM file.")
                else:
                    precip = f["Grid/precipitationCal"][:]
                    valid_mask = (precip >= 0.0) & (precip < 500.0)
                    valid_pct = (np.count_nonzero(valid_mask) / precip.size) * 100.0
                    report["shape"] = list(precip.shape)
                    report["valid_pct"] = round(valid_pct, 2)
        except Exception as e:
            report["status"] = "INVALID"
            report["errors"].append(str(e))

        if report["errors"]:
            report["status"] = "INVALID"
        return report
""",

    "src/data/insat_loader.py": """from pathlib import Path
import h5py
import numpy as np
import xarray as xr
from src.utils.geo_utils import generate_monsoon_target_grid, regrid_to_reference

class INSAT3DLoader:
    CENTRAL_WAVELENGTHS = {"TIR1": 10.8, "TIR2": 12.0, "WV": 6.7, "MIR": 3.9}
    C1 = 1.191042e8
    C2 = 1.4387752

    def __init__(self, target_resolution: float = 0.04):
        self.target_lons, self.target_lats = generate_monsoon_target_grid(resolution=target_resolution)

    def _convert_radiance_to_bt(self, radiance: np.ndarray, channel: str) -> np.ndarray:
        nu = 10000.0 / self.CENTRAL_WAVELENGTHS[channel]
        valid_mask = radiance > 0.0
        bt = np.full_like(radiance, np.nan, dtype=np.float32)
        arg = 1.0 + (self.C1 * (nu ** 3)) / radiance[valid_mask]
        bt[valid_mask] = (self.C2 * nu) / np.log(arg)
        return bt

    def load_and_calibrate(self, filepath: Path) -> xr.Dataset:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Real INSAT-3D file missing: {filepath}")

        calibrated = {}
        with h5py.File(filepath, "r") as f:
            lat = f["Latitude"][:] if "Latitude" in f else None
            lon = f["Longitude"][:] if "Longitude" in f else None
            channel_keys = {
                "TIR1": [k for k in f.keys() if "IMG_TIR1" in k][0],
                "TIR2": [k for k in f.keys() if "IMG_TIR2" in k][0],
                "WV": [k for k in f.keys() if "IMG_WV" in k][0],
                "MIR": [k for k in f.keys() if "IMG_MIR" in k][0]
            }
            for ch, key in channel_keys.items():
                raw_data = f[key][:].astype(np.float32)
                scale = f[key].attrs.get("scale_factor", 1.0)
                offset = f[key].attrs.get("add_offset", 0.0)
                calibrated[ch] = self._convert_radiance_to_bt(raw_data * scale + offset, ch)

        if lat is not None and lon is not None:
            source_lats = lat[:, 0] if lat.ndim == 2 else lat
            source_lons = lon[0, :] if lon.ndim == 2 else lon
        else:
            source_lats = np.linspace(45.0, -45.0, calibrated["TIR1"].shape[0])
            source_lons = np.linspace(30.0, 120.0, calibrated["TIR1"].shape[1])

        ds_vars = {}
        for ch, data in calibrated.items():
            da = xr.DataArray(data, coords=[("latitude", source_lats), ("longitude", source_lons)], name=f"BT_{ch}")
            ds_vars[f"BT_{ch}"] = regrid_to_reference(da, self.target_lons, self.target_lats)

        ds = xr.Dataset(ds_vars)
        ds["BTD_SplitWindow"] = ds["BT_TIR1"] - ds["BT_TIR2"]
        ds["BTD_Overshoot"] = ds["BT_WV"] - ds["BT_TIR1"]
        return ds
""",

    "src/data/gpm_loader.py": """from pathlib import Path
import h5py
import numpy as np
import xarray as xr
from src.utils.geo_utils import generate_monsoon_target_grid, regrid_to_reference

class GPMLoader:
    def __init__(self, target_resolution: float = 0.04, heavy_rain_threshold: float = 7.5):
        self.target_lons, self.target_lats = generate_monsoon_target_grid(resolution=target_resolution)
        self.heavy_rain_threshold = heavy_rain_threshold

    def load_precipitation(self, filepath: Path) -> xr.Dataset:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Real GPM IMERG file missing: {filepath}")

        with h5py.File(filepath, "r") as f:
            precip = f["Grid/precipitationCal"][:].astype(np.float32)
            lons = f["Grid/lon"][:]
            lats = f["Grid/lat"][:]

        if precip.ndim == 3:
            precip = precip[0]
        if precip.shape == (len(lons), len(lats)):
            precip = np.transpose(precip, (1, 0))

        precip[precip < 0.0] = np.nan
        da = xr.DataArray(precip, coords=[("latitude", lats), ("longitude", lons)], name="precipitation_rate_mm_hr")
        regridded_precip = regrid_to_reference(da, self.target_lons, self.target_lats)
        binary_mask = xr.where(regridded_precip >= self.heavy_rain_threshold, 1.0, 0.0)

        return xr.Dataset({"qpe_rate": regridded_precip, "heavy_rain_mask": binary_mask})
""",

    "src/data/dataset.py": """from pathlib import Path
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
""",

    "tests/test_real_data_pipeline.py": """from pathlib import Path
import numpy as np
import xarray as xr
import pytest
from src.utils.geo_utils import generate_monsoon_target_grid, regrid_to_reference
from src.data.dataset import split_chronological_sequences
from src.data.validator import DatasetValidator

def test_monsoon_target_grid_generation():
    lons, lats = generate_monsoon_target_grid(min_lon=65.0, min_lat=5.0, max_lon=95.0, max_lat=38.0, resolution=0.04)
    assert lons[0] == 65.0
    assert lons[-1] == 95.0
    assert lats[0] == 5.0
    assert lats[-1] == 38.0
    assert len(lons) == 751
    assert len(lats) == 826

def test_regrid_interpolation_shapes():
    target_lons, target_lats = generate_monsoon_target_grid(min_lon=70.0, min_lat=10.0, max_lon=75.0, max_lat=15.0, resolution=0.1)
    src_lons = np.arange(68.0, 78.0, 0.25)
    src_lats = np.arange(8.0, 18.0, 0.25)
    src_data = np.random.uniform(200.0, 300.0, size=(len(src_lats), len(src_lons)))

    da = xr.DataArray(src_data, coords=[("latitude", src_lats), ("longitude", src_lons)], name="test_var")
    regridded = regrid_to_reference(da, target_lons, target_lats)
    assert regridded.shape == (len(target_lats), len(target_lons))

def test_zero_leakage_chronological_split():
    mock_timestamps = [f"2023-06-01T{h:02d}:00:00" for h in range(100)]
    splits = split_chronological_sequences(mock_timestamps, train_ratio=0.70, val_ratio=0.15)
    assert len(set(splits["train_indices"]).intersection(set(splits["val_indices"]))) == 0
    assert len(set(splits["train_indices"]).intersection(set(splits["test_indices"]))) == 0
    assert max(splits["train_indices"]) < min(splits["val_indices"])

def test_validator_fails_on_missing_file():
    validator = DatasetValidator()
    with pytest.raises(FileNotFoundError):
        validator.validate_insat_file(Path("non_existent_insat.h5"))
"""
}

for path_str, content in files.items():
    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(content)
    print(f"Created: {path_str}")

print("\nAll pipeline files successfully written to src/ and tests/!")
