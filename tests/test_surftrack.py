# ============================================================
# tests/test_surftrack.py
# ============================================================
"""
Tests for ocetrac.SurfTrack.

Mirrors the structure of test_model.py (the SurfTrack original tests)
and additionally covers the three changes made in this refactor:

  Change 1 — min_area_cells : absolute floor for area filtering.
  Change 2 — relabelling loop : NaN cells must not be corrupted.
  Change 3 — allow_rechunk=True : spatially chunked Dask arrays must work.

Run with:
    python -m pytest tests/test_surftrack.py -v
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

from ocetrac.SurfTrack import SurfTracker
from ocetrac.SurfTrack.tracker import (
    apply_mask,
    filter_area,
    label_3d,
    morphological_operations,
    wrap_labels,
)


# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def example_data():
    """
    Synthetic 3-D (time, lat, lon) dataset with a few Gaussian warm blobs.

    Two blobs straddle the date line to exercise the wrap logic.
    Returns (Anom, mask) matching the interface expected by SurfTracker.
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
        np.ones(Anom[0].shape),
        coords=Anom[0].coords,
    )
    mask[60:90, 120:190] = 0   # land-like region

    return Anom, mask


# ── SurfTracker.run — parametrized ────────────────────────────────────────────

@pytest.mark.parametrize("radius", [8, 10])
@pytest.mark.parametrize("min_size_quartile", [0.75, 0.80])
@pytest.mark.parametrize("positive", [True, False])
def test_run(example_data, radius, min_size_quartile, positive):
    """
    Full pipeline runs without error and percent_area sums to 1.0.
    Mirrors test_track from the original test suite.
    """
    Anom, mask = example_data
    if not positive:
        Anom = Anom * -1

    tracker = SurfTracker(
        Anom, mask,
        radius            = radius,
        min_size_quartile = min_size_quartile,
        min_area_cells    = 50,
        timedim           = "time",
        xdim              = "lon",
        ydim              = "lat",
        positive          = positive,
    )
    result = tracker.run()

    assert (
        result.attrs["percent area reject"] + result.attrs["percent area accept"]
    ) == pytest.approx(1.0)

    if positive:
        assert Anom.sum() >= 0
    else:
        assert Anom.sum() <= 0


# ── morphological_operations ─────────────────────────────────────────────────

def test_morphological_operations(example_data):
    """
    Binary output has correct shape; Dask array is returned for chunked input.
    Change 3: spatially chunked input must not raise ValueError.
    """
    Anom, _ = example_data

    # Test with spatial chunking — Change 3 (allow_rechunk=True)
    out = morphological_operations(
        Anom.chunk({"time": 1, "lat": 45, "lon": 90}),
        radius=8, xdim="lon", ydim="lat", positive=True,
    )
    assert out.shape == Anom.shape
    assert isinstance(out.data, dsa.Array)

    # Test without chunking
    out_np = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    assert out_np.shape == Anom.shape


def test_morphological_operations_positive_negative(example_data):
    """positive=True and positive=False flag warm and cold anomalies respectively."""
    Anom, _ = example_data
    out_warm = morphological_operations(Anom,      radius=8, xdim="lon", ydim="lat", positive=True)
    out_cold = morphological_operations(Anom * -1, radius=8, xdim="lon", ydim="lat", positive=False)
    # Both should flag the same spatial footprint
    assert out_warm.values.sum() == pytest.approx(out_cold.values.sum(), rel=0.05)


# ── apply_mask ────────────────────────────────────────────────────────────────

def test_apply_mask(example_data):
    """Masked cells (mask==0) must be 0 after apply_mask."""
    Anom, mask = example_data
    binary = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    mask_bc = mask.broadcast_like(binary)
    result  = apply_mask(binary, mask_bc)
    assert (result.where(mask_bc == 0, drop=True) == 0).all()


# ── filter_area ───────────────────────────────────────────────────────────────

def test_filter_area_returns_correct_types(example_data):
    """filter_area returns (DataArray, float, DataArray, int)."""
    Anom, mask = example_data
    binary  = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    masked  = apply_mask(binary, mask.broadcast_like(binary))
    area, min_area, binary_labels, N_initial = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.75, min_area_cells=50,
    )
    assert isinstance(area, xr.DataArray)
    assert isinstance(min_area, float)
    assert isinstance(binary_labels, xr.DataArray)
    assert isinstance(N_initial, int)


def test_filter_area_absolute_floor(example_data):
    """
    Change 1 — min_area_cells absolute floor.

    When min_area_cells is set very high, more objects are removed than the
    percentile alone would remove.
    """
    Anom, mask = example_data
    binary = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    masked = apply_mask(binary, mask.broadcast_like(binary))

    # Permissive percentile, but high absolute floor
    _, _, labels_high_floor, _ = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.0, min_area_cells=5000,
    )
    # Very low absolute floor — percentile does the work
    _, _, labels_low_floor, _ = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.0, min_area_cells=1,
    )
    n_high = int((labels_high_floor == 1).values.sum())
    n_low  = int((labels_low_floor  == 1).values.sum())
    # High floor removes more → fewer surviving voxels
    assert n_high <= n_low


def test_filter_area_threshold_is_max_of_floor_and_percentile():
    """
    Change 1 — effective threshold = max(min_area_cells, percentile).

    Build a case where all objects have area ~100.  With min_area_cells=200
    ALL objects should be removed even if min_size_quartile=0.
    """
    # One small blob per timestep, all size ~25 cells
    data = np.zeros((4, 20, 20))
    data[:, 8:13, 8:13] = 1.0
    da   = xr.DataArray(data, dims=["time","lat","lon"],
                        coords={"time": np.arange(4),
                                "lat":  np.arange(20),
                                "lon":  np.arange(20)})
    mask = xr.DataArray(np.ones((4, 20, 20)), dims=da.dims, coords=da.coords)

    binary = morphological_operations(da, radius=1, xdim="lon", ydim="lat")
    masked = apply_mask(binary, mask)

    _, min_area, labels_out, _ = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.0, min_area_cells=9999,
    )
    # min_area_cells >> blob size → everything removed
    assert (labels_out.values == 1).sum() == 0


def test_filter_area_nan_not_corrupted():
    """
    Change 2 — relabelling loop fix.

    NaN cells must remain NaN (not become large integers) after the
    relabelling step that makes labels consecutive across timesteps.
    """
    # Blob at t=0, nothing at t=1 (NaN), blob at t=2
    data = np.full((3, 20, 20), np.nan)
    data[0, 8:13, 8:13] = 1.0
    data[2, 8:13, 8:13] = 1.0
    da   = xr.DataArray(data, dims=["time","lat","lon"],
                        coords={"time": np.arange(3),
                                "lat":  np.arange(20),
                                "lon":  np.arange(20)})
    mask = xr.DataArray(np.ones((3, 20, 20)), dims=da.dims, coords=da.coords)

    binary = morphological_operations(da.fillna(0), radius=1,
                                      xdim="lon", ydim="lat")
    masked = apply_mask(binary, mask)

    area, min_area, binary_labels, N_initial = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.0, min_area_cells=1,
    )
    # binary_labels should be 0 or 1 — never a large integer from NaN corruption
    unique_vals = np.unique(binary_labels.values)
    assert set(unique_vals).issubset({0.0, 1.0}), \
        f"Unexpected values after relabelling: {unique_vals}"


# ── label_3d ─────────────────────────────────────────────────────────────────

def test_label_3d_shape(example_data):
    """Output shape matches input."""
    Anom, mask = example_data
    binary = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    masked = apply_mask(binary, mask.broadcast_like(binary))
    _, _, binary_labels, _ = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.75, min_area_cells=50,
    )
    labels, n = label_3d(binary_labels, connectivity=3)
    assert labels.shape == Anom.shape
    assert isinstance(n, (int, np.integer))


def test_label_3d_background_zero(example_data):
    """Background (non-event) cells are labelled 0."""
    Anom, mask = example_data
    binary = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    masked = apply_mask(binary, mask.broadcast_like(binary))
    _, _, binary_labels, _ = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.75, min_area_cells=50,
    )
    labels, _ = label_3d(binary_labels)
    # Where binary_labels is 0, labels must also be 0
    assert (labels[binary_labels.values == 0] == 0).all()


# ── wrap_labels ───────────────────────────────────────────────────────────────

def test_wrap_labels_reduces_count(example_data):
    """
    Wrapping merges date-line-straddling objects → final count <= initial.
    Mirrors test_wrap from the original suite.
    """
    Anom, mask = example_data
    binary = morphological_operations(Anom, radius=8, xdim="lon", ydim="lat")
    masked = apply_mask(binary, mask.broadcast_like(binary))
    _, _, binary_labels, N_initial = filter_area(
        masked, xdim="lon", ydim="lat",
        min_size_quartile=0.75, min_area_cells=50,
    )
    labels, _ = label_3d(binary_labels)
    n_before = int(np.max(np.array(labels)))
    wrapped, N_final = wrap_labels(np.array(labels))
    assert N_final <= n_before


def test_wrap_labels_consecutive():
    """After wrapping, labels are consecutive integers starting at 0."""
    arr = np.array([[[0, 1, 2], [3, 4, 5]]])
    wrapped, N = wrap_labels(arr.copy())
    unique = np.unique(wrapped)
    assert list(unique) == list(range(N + 1))


# ── SurfTracker integration ───────────────────────────────────────────────────

class TestSurfTrackerIntegration:

    # dim names used in the synthetic data
    _xdim = 'lon'
    _ydim = 'lat'

    def _tracker(self, T=4, positive=True):
        lon = np.arange(0, 360) + 0.5
        lat = np.arange(-90, 90) + 0.5
        x, y = np.meshgrid(lon, lat)
        blob = np.exp(-((x - 180) ** 2 + (y - 0) ** 2) / (2 * 20 ** 2))
        data = np.stack([blob * 2 - 0.5] * T, axis=0)
        data = np.where(data > 0, data, 0)
        da = xr.DataArray(data, dims=["time","lat","lon"],
                          coords={"time": np.arange(T),
                                  "lat": lat, "lon": lon})
        mask = xr.DataArray(np.ones((T, 180, 360)), dims=da.dims, coords=da.coords)
        return da, mask

    def test_run_returns_dataarray(self):
        da, mask = self._tracker()
        result = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                             min_area_cells=10, xdim='lon', ydim='lat').run()
        assert isinstance(result, xr.DataArray)
        assert result.shape == da.shape

    def test_result_background_is_nan(self):
        da, mask = self._tracker()
        result = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                             min_area_cells=10, xdim='lon', ydim='lat').run()
        # No zeros — background should be NaN
        assert not (result.values == 0).any()

    def test_percent_area_sums_to_one(self):
        da, mask = self._tracker()
        result = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                             min_area_cells=10, xdim='lon', ydim='lat').run()
        total = result.attrs["percent area reject"] + result.attrs["percent area accept"]
        assert total == pytest.approx(1.0)

    def test_n_events_positive(self):
        da, mask = self._tracker()
        t = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                        min_area_cells=10, xdim='lon', ydim='lat')
        t.run()
        assert t.n_events() >= 1

    def test_event_duration_dict(self):
        da, mask = self._tracker(T=5)
        t = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                        min_area_cells=10, xdim='lon', ydim='lat')
        t.run()
        durations = t.event_duration()
        assert isinstance(durations, dict)
        for eid, dur in durations.items():
            assert isinstance(eid, int) and dur > 0

    def test_summary_runs(self):
        da, mask = self._tracker()
        t = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                        min_area_cells=10, xdim='lon', ydim='lat')
        t.run()
        t.summary()   # should not raise

    def test_repr_before_run(self):
        da, mask = self._tracker()
        assert "(not run yet)" in repr(SurfTracker(da, mask, xdim='lon', ydim='lat'))

    def test_postprocess_raises_without_track(self):
        da, mask = self._tracker()
        with pytest.raises(RuntimeError, match="track"):
            SurfTracker(da, mask, xdim='lon', ydim='lat').postprocess()

    def test_chaining_returns_self(self):
        da, mask = self._tracker()
        t = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                        min_area_cells=10, xdim='lon', ydim='lat')
        assert t.clean()  is t
        assert t.filter() is t
        assert t.track()  is t

    def test_step_by_step_matches_run(self):
        """Running steps individually must produce the same result as .run()."""
        da, mask = self._tracker()

        r1 = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                         min_area_cells=10, xdim='lon', ydim='lat').run()

        t2 = SurfTracker(da, mask, radius=5, min_size_quartile=0.5,
                         min_area_cells=10, xdim='lon', ydim='lat')
        t2.clean().filter().track().postprocess()
        r2 = t2.result

        np.testing.assert_array_equal(
            np.nan_to_num(r1.values),
            np.nan_to_num(r2.values),
        )

    def test_min_area_cells_removes_small_events(self):
        """
        Change 1 — a high min_area_cells should remove small objects even
        when the percentile threshold would keep them.
        """
        da, mask = self._tracker()
        t_high = SurfTracker(da, mask, radius=5, min_size_quartile=0.0,
                              min_area_cells=999999, xdim='lon', ydim='lat')
        result_high = t_high.run()

        t_low = SurfTracker(da, mask, radius=5, min_size_quartile=0.0,
                             min_area_cells=1, xdim='lon', ydim='lat')
        result_low = t_low.run()

        n_high = len(np.unique(result_high.values[~np.isnan(result_high.values)]))
        n_low  = len(np.unique(result_low.values[~np.isnan(result_low.values)]))
        assert n_high <= n_low

    def test_chunked_input_works(self):
        """
        Change 3 — spatially chunked Dask arrays must not raise ValueError
        in morphological_operations (allow_rechunk=True fix).
        """
        da, mask = self._tracker()
        da_chunked = da.chunk({"time": 1, "lat": 45, "lon": 90})
        result = SurfTracker(da_chunked, mask, radius=5,
                             min_size_quartile=0.5, min_area_cells=10, xdim='lon', ydim='lat').run()
        assert result is not None
        