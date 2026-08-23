from typing import Tuple
import numpy as np
import xarray as xr

def generate_monsoon_target_grid(
    min_lon: float = 65.0,
    min_lat: float = 5.0,
    max_lon: float = 95.0,
    max_lat: float = 38.0,
    resolution: float = 0.04
) -> Tuple[np.ndarray, np.ndarray]:
    num_lons = int(round((max_lon - min_lon) / resolution)) + 1
    num_lats = int(round((max_lat - min_lat) / resolution)) + 1
    lons = np.round(np.linspace(min_lon, max_lon, num_lons), 4)
    lats = np.round(np.linspace(min_lat, max_lat, num_lats), 4)
    return lons, lats

def regrid_to_reference(
    source_da: xr.DataArray,
    target_lons: np.ndarray,
    target_lats: np.ndarray,
    method: str = "linear"
) -> xr.DataArray:
    lat_dim = [d for d in source_da.dims if d.lower() in ["lat", "latitude"]][0]
    lon_dim = [d for d in source_da.dims if d.lower() in ["lon", "longitude"]][0]

    reindexed = source_da.rename({lat_dim: "latitude", lon_dim: "longitude"})
    
    if reindexed.latitude[0] > reindexed.latitude[-1]:
        reindexed = reindexed.reindex(latitude=reindexed.latitude[::-1])
    if reindexed.longitude[0] > reindexed.longitude[-1]:
        reindexed = reindexed.reindex(longitude=reindexed.longitude[::-1])

    return reindexed.interp(latitude=target_lats, longitude=target_lons, method=method)
