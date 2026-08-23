from pathlib import Path
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
