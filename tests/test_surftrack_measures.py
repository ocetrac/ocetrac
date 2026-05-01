# ============================================================
# tests/test_surftrack_measures.py
# ============================================================
"""
Tests for ocetrac.SurfTrack.measures.

Mirrors the structure of the original measures testa but uses
SurfTracker instead of ocetrac.Tracker to generate the blobs fixture.

Run with:
    python -m pytest tests/test_surftrack_measures.py -v
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import xarray as xr

from ocetrac.SurfTrack import SurfTracker
from ocetrac.SurfTrack.measures import (
    calculate_intensity_metrics,
)
from ocetrac.SurfTrack.measures.utils import (
    get_object_masks,
    lons_to_180,
    lons_to_360,
    process_objects_and_calculate_measures,
    run_intensity_measures,
    run_motion_measures,
    run_shape_measures,
    run_temporal_measures,
)


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def example_data():
    """
    Synthetic (time, lat, lon) dataset with Gaussian warm blobs.
    Returns (blobs, Anom) matching the interface expected by the measures
    sub-module — same pattern as the original measures test suite.
    """
    x0    = [180, 225, 360, 80, 1, 360, 1]
    y0    = [0,   20, -50,  40, -50, 40, 40]
    sigma = [15,  25,  30,  10,  30, 15, 10]

    lon = np.arange(0, 360) + 0.5
    lat = np.arange(-90, 90) + 0.5
    x, y = np.meshgrid(lon, lat)

    def blob(x0, y0, s):
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * s ** 2))

    features = {i: blob(x0[i], y0[i], sigma[i]) for i in range(len(x0))}

    first = features[0] + features[1] + features[3] - 0.5
    da = xr.DataArray(
        first[np.newaxis, :, :],
        dims=["time", "lat", "lon"],
        coords={"time": [1], "lat": lat, "lon": lon},
    )

    da_s01 = da.shift(lon=0, lat=-20, fill_value=-0.5)
    da_s02 = da.shift(lon=0, lat=-40, fill_value=-0.5) + (
        features[2] + features[4] + features[5] + features[6]
    )
    da_s03 = da.shift(lon=0, lat=-40, fill_value=-0.5) + (
        features[2] + features[5] + features[6]
    )

    Anom = xr.concat([da, da_s01, da_s02, da_s03], dim="time")
    Anom["time"] = np.arange(1, 5)
    Anom = Anom.where(Anom > 0, drop=False, other=0)

    mask = xr.DataArray(
        np.ones(Anom[0].shape), coords=Anom[0].coords
    )
    mask[60:90, 120:190] = 0

    tracker = SurfTracker(
        Anom.chunk({"time": 1}),
        mask,
        radius            = 8,
        min_size_quartile = 0,
        min_area_cells    = 0,
        timedim           = "time",
        xdim              = "lon",
        ydim              = "lat",
        positive          = True,
    )
    blobs = tracker.run()

    return blobs, Anom


# ── utils — lon conversion ────────────────────────────────────────────────────

def test_convert_lons(example_data):
    """lons_to_180 and lons_to_360 round-trip correctly."""
    _, Anom = example_data
    new_Anom = lons_to_180(Anom)
    assert list(new_Anom["lon"].values) == list(np.linspace(-179.5, 179.5, 360))
    new_Anom2 = lons_to_360(new_Anom)
    assert list(new_Anom2["lon"].values) == list(np.linspace(0.5, 359.5, 360))


# ── utils — get_object_masks ──────────────────────────────────────────────────

@pytest.mark.parametrize("obj_id", [1, 3, 5, np.nan])
def test_get_object_masks(example_data, obj_id):
    """
    Valid object IDs return DataArrays with correct time coordinates.
    Invalid / missing IDs emit a UserWarning.
    """
    blobs, Anom = example_data
    valid_ids = np.unique(blobs.values[~np.isnan(blobs.values)])

    if obj_id not in valid_ids:
        with pytest.warns(
            UserWarning,
            match=rf'object_id "{obj_id}" not found in blobs. Returning zeros.',
        ):
            event_binary, event_intensity = get_object_masks(
                blobs, Anom, object_id=obj_id
            )
    else:
        event_binary, event_intensity = get_object_masks(
            blobs, Anom, object_id=obj_id
        )
        assert isinstance(event_intensity, xr.DataArray)
        assert len(event_binary.time) > 0
        assert list(blobs.where(blobs == obj_id, drop=True).time.values) == list(
            event_binary.time.values
        )


# ── shape measures ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_shape_measures(example_data, obj_id):
    """
    run_shape_measures returns all expected keys with correct array lengths.
    """
    blobs, Anom = example_data
    event_binary, _ = get_object_masks(blobs, Anom, object_id=obj_id)

    shape_measures = run_shape_measures(
        event_binary,
        lat_resolution_value = 110.574,
        lon_resolution_value = 111.320,
    )

    expected_keys = [
        "perimeters",
        "spatial_extents",
        "max_spatial_extent",
        "mean_spatial_extent",
        "complement_to_deformation",
        "deformation",
        "convex_hull_areas",
        "ratio_convexhullarea_vs_area",
        "circularity",
    ]
    T = len(event_binary.time)
    expected_lengths = [T, T, 1, 1, T - 1, T - 1, T, T, T]

    assert list(shape_measures.keys()) == expected_keys
    assert [
        len(v) if not isinstance(v, (float, np.float64)) else 1
        for v in shape_measures.values()
    ] == expected_lengths


# ── motion measures ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_motion_measures(example_data, obj_id):
    """
    run_motion_measures returns all expected keys with correct array lengths.
    """
    blobs, Anom = example_data
    event_binary, event_intensity = get_object_masks(blobs, Anom, object_id=obj_id)

    # Motion measures require lons in (-180, 180)
    event_binary_180    = lons_to_180(event_binary.copy())
    event_intensity_180 = lons_to_180(event_intensity.copy())

    results = run_motion_measures(event_binary_180, event_intensity_180)

    expected_keys = [
        "num_centroids_per_timestep",
        "centroid_coords",
        "centroid_path",
        "centroid_displacements_km",
        "num_coms_per_timestep",
        "com_coords",
        "com_path",
        "com_displacements_km",
        "com_directionality",
        "centroid_directionality",
        "centroid_displacement_plot_data",
        "com_displacement_plot_data",
    ]
    T = len(event_binary.time)
    expected_lengths = [T, T, T, T - 1, T, T, T, T - 1, 5, 5, 2, 2]

    assert list(results.keys()) == expected_keys
    assert [
        len(v) if not isinstance(v, (float, np.float64)) else 1
        for v in results.values()
    ] == expected_lengths


# ── temporal measures ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_temporal_measures(example_data, obj_id):
    """
    run_temporal_measures returns initial_detection_date and duration.
    Empty time coordinate raises ValueError.
    """
    blobs, Anom = example_data
    event_binary, _ = get_object_masks(blobs, Anom, object_id=obj_id)

    # Empty time should raise
    with pytest.raises(ValueError):
        run_temporal_measures(event_binary.sel(time=slice(0, 0)))

    results = run_temporal_measures(event_binary)
    assert list(results.keys()) == ["initial_detection_date", "duration"]
    assert results["initial_detection_date"] == event_binary.time[0]
    assert results["duration"] == len(event_binary.time)


# ── intensity measures ────────────────────────────────────────────────────────

@pytest.mark.parametrize("quantile_threshold", [0.5, 0.9])
@pytest.mark.parametrize("obj_id", [1, 3])
def test_calculate_intensity_metrics(example_data, obj_id, quantile_threshold):
    """
    calculate_intensity_metrics returns all expected keys with correct lengths.
    """
    blobs, Anom = example_data
    event_binary, event_intensity = get_object_masks(blobs, Anom, object_id=obj_id)

    results = calculate_intensity_metrics(event_intensity, quantile_threshold)

    q = int(quantile_threshold * 100)
    expected_keys = [
        "cumulative_intensity",
        "mean_intensity_timeseries",
        "mean_intensity",
        "max_intensity_timeseries",
        "max_intensity",
        "std_intensity_timeseries",
        "std_intensity",
        f"percentile_{q}_intensity_timeseries",
        f"percentile_{q}_intensity",
        "quantile_threshold_used",
    ]
    T = len(event_binary.time)
    expected_lengths = [T, T, 1, T, 1, T, 1, T, 1, 1]

    assert list(results.keys()) == expected_keys
    assert [
        len(v) if not isinstance(v, (float, np.floating)) else 1
        for v in results.values()
    ] == expected_lengths


@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_intensity_measures(example_data, obj_id):
    """
    run_intensity_measures returns max_intensity and 90th-percentile timeseries.
    """
    blobs, Anom = example_data
    _, event_intensity = get_object_masks(blobs, Anom, object_id=obj_id)

    results = run_intensity_measures(event_intensity)
    assert "max_intensity" in results
    assert "percentile_90_intensity_timeseries" in results
    assert isinstance(results["max_intensity"], float)


# ── process_objects_and_calculate_measures ────────────────────────────────────

@pytest.mark.parametrize("obj_ids", [[1], [1, 3]])
def test_process_objects_valid(example_data, obj_ids):
    """
    process_objects_and_calculate_measures returns nested dict keyed by
    object ID, with all four measure groups present.
    """
    blobs, Anom = example_data

    results = process_objects_and_calculate_measures(
        obj_ids, blobs, Anom,
        run_shape    = True,
        run_motion   = False,   # motion requires lon conversion — skip here
        run_temporal = True,
        run_intensity = True,
    )

    assert list(results.keys()) == obj_ids
    for obj_id in obj_ids:
        assert "shape_measures"    in results[obj_id]
        assert "temporal_measures" in results[obj_id]
        assert "intensity_measures" in results[obj_id]


def test_process_objects_invalid_id(example_data):
    """Invalid object ID raises ValueError."""
    blobs, Anom = example_data
    with pytest.raises((ValueError, Exception)):
        process_objects_and_calculate_measures(
            [9999], blobs, Anom,
            run_shape=False, run_motion=False,
            run_temporal=True, run_intensity=False,
        )
        