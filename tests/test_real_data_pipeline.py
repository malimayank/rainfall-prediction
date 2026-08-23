from pathlib import Path
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
