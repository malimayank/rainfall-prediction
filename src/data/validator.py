from pathlib import Path
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
