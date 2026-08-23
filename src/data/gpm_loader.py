from pathlib import Path
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
