# ============================================================
# tests/test_surftrack.py
# ============================================================
"""
Tests for ocetrac.SurfTrack.

Covers the full pipeline plus four specific changes made during refactoring:

  Change 1 — min_area_cells : absolute floor for area filtering.
  Change 2 — relabelling loop : NaN cells must not be corrupted.
  Change 3 — allow_rechunk=True : spatially chunked Dask arrays must work.
  Change 4 - contain_thresh edge case:
                0.0  any shared pixel merges blobs (most permissive)
                1.0  one blob must be fully contained in the other (strictest)

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
    label_temporal_neighbor,
    morphological_operations,
    wrap_labels,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_da(frames, xdim="lon", ydim="lat"):
    """Stack a list of 2-D arrays into a (time, ydim, xdim) DataArray."""
    arr = np.stack(frames, axis=0).astype(float)
    T, Y, X = arr.shape
    return xr.DataArray(
        arr,
        dims=["time", ydim, xdim],
        coords={
            "time": np.arange(T),
            ydim:   np.arange(Y),
            xdim:   np.arange(X),
        },
    )

# ── shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def example_data():
    """
    Synthetic (time, lat, lon) dataset built from Gaussian blobs.
    Two blobs straddle the date line to exercise the wrap logic.
    Returns (Anom, mask) as expected by SurfTracker.
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


# ── contain_thresh tests ──────────────────────────────────────────────────

@pytest.fixture
def single_pixel_overlap():
    """
    Two 2x2 blobs that share one column.
 
        t=0  cols 1-2  (4 cells)
        t=1  cols 2-3  (4 cells)
 
    Overlap = 2 cells, so max(|A∩B|/|A|, |A∩B|/|B|) = 0.5 exactly.
    Merges at thresh <= 0.5, stays separate at thresh > 0.5.
    """
    frame0=np.zeros((5,6))
    frame0[1:3,1:3] = 1

    frame1=np.zeros((5,6))
    frame1[1:3,2:4]=1
    return _make_da([frame0, frame1])

@pytest.fixture
def full_containment():
    """
    Small blob at t=1 sits entirely inside larger blob at t=0.
 
        t=0  5x5 square (25 cells)
        t=1  3x3 square centred inside t=0 (9 cells)
 
    |A∩B|/|B| = 9/9 = 1.0, so max containment = 1.0.
    Merges at any thresh including 1.0.
    """
    frame0=np.zeros((10,10))
    frame0[2:7,2:7]=1

    frame1=np.zeros((10,10))
    frame1[3:6,3:6]=1
    return _make_da([frame0, frame1])

@pytest.fixture
def no_overlap():
    """Two blobs with zero shared pixels."""
    f0 = np.zeros((5, 10)); f0[1:4, 1:4] = 1
    f1 = np.zeros((5, 10)); f1[1:4, 6:9] = 1
    return _make_da([f0, f1])
@pytest.fixture
def three_frames_chain():
    """
    A 2x2 blob shifts one column right each timestep.
    Adjacent-frame containment = 0.5, so:
        thresh=0.0  -> all three frames are one event
        thresh=1.0  -> each frame is its own event
    """
    frames = []
    for col_start in [1, 2, 3]:
        f = np.zeros((5, 8))
        f[1:3, col_start:col_start + 2] = 1   # 2×2 = 4 cells
        frames.append(f)
    return _make_da(frames)

# ── SurfTracker.run — across both methods ────────────────────────────────────────────

@pytest.mark.parametrize("method", ["3d", "temporal_neighbor"])
@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("min_size_quartile", [0.75, 0.80])
@pytest.mark.parametrize("radius", [8, 10])
def test_run(example_data, method, radius, min_size_quartile, positive):
    """
    Full pipeline runs without error and percent_area sums to 1.0.
    Parametrized over both labelling methods.
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
        method            = method,
        contain_thresh    = 0.0,
    )
    result = tracker.run()

    assert (
        result.attrs["percent area reject"] + result.attrs["percent area accept"]
    ) == pytest.approx(1.0)
    assert result.attrs["method"] == method
    
    if positive:
        assert Anom.sum() >= 0
    else:
        assert Anom.sum() <= 0


# ── morphological_operations ─────────────────────────────────────────────────

def test_morphological_operations(example_data):
    """Output shape matches input; chunked input returns a Dask array."""
    Anom, _ = example_data

    out = morphological_operations(
        Anom.chunk({"time": 1, "lat": 45, "lon": 90}),
        radius=8, xdim="lon", ydim="lat", positive=True,
    )
    assert out.shape == Anom.shape
    assert isinstance(out.data, dsa.Array)

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
    """Change 1 — a high min_area_cells removes more objects than the percentile alone."""
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
    """Change 1 — when min_area_cells >> blob size, everything is removed."""
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
    assert (labels_out.values == 1).sum() == 0


def test_filter_area_nan_not_corrupted():
    """Change 2 — NaN cells must stay 0/1 and never become large integers."""
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
    unique_vals = np.unique(binary_labels.values)
    assert set(unique_vals).issubset({0.0, 1.0}), \
        f"Unexpected values after relabelling: {unique_vals}"


# ── label_3d ─────────────────────────────────────────────────────────────────

def test_label_3d_shape(example_data):
    """label_3d output shape matches input shape."""
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
    """Cells that are 0 in the binary input must be 0 in the label output."""
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
    """Wrapping merges date-line objects so the final count is <= the initial count."""
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
    """Labels after wrapping are consecutive integers starting at 0."""
    arr = np.array([[[0, 1, 2], [3, 4, 5]]])
    wrapped, N = wrap_labels(arr.copy())
    unique = np.unique(wrapped)
    assert list(unique) == list(range(N + 1))

# ── label_temporal_neighbor - contain_thresh = 0.0 ───────────────────────────────────────────────────────────────

class TestContainThresh0:
    """thresh=0.0"""
 
    def test_partial_overlap_merges(self, single_pixel_overlap):
        # max containment = 0.5 >= 0.0, so the two blobs should merge
        _, n = label_temporal_neighbor(single_pixel_overlap, contain_thresh=0.0)
        assert n == 1, f"Expected 1 event, got {n}"
 
    def test_no_overlap_stays_separate(self, no_overlap):
        # zero shared pixels — cannot merge regardless of thresh
        _, n = label_temporal_neighbor(no_overlap, contain_thresh=0.0)
        assert n == 2, f"Expected 2 events, got {n}"
 
    def test_chain_is_one_event(self, three_frames_chain):
        # each adjacent pair overlaps, so all three frames chain into one event
        _, n = label_temporal_neighbor(three_frames_chain, contain_thresh=0.0)
        assert n == 1, f"Expected 1 chained event, got {n}"
 
    def test_full_containment_merges(self, full_containment):
        # full containment trivially satisfies thresh=0.0
        _, n = label_temporal_neighbor(full_containment, contain_thresh=0.0)
        assert n == 1, f"Expected 1 event, got {n}"
 
    def test_background_stays_zero(self, single_pixel_overlap):
        # labelled cells get a positive integer; background stays 0
        labels, _ = label_temporal_neighbor(single_pixel_overlap, contain_thresh=0.0)
        assert (labels[single_pixel_overlap.values > 0] > 0).all()
        assert (labels[single_pixel_overlap.values == 0] == 0).all()

# ── label_temporal_neighbor - contain_thresh = 1.0 ───────────────────────────────────────────────────────────────

class TestContainThresh1:
    """thresh=1.0"""
 
    def test_partial_overlap_stays_separate(self, single_pixel_overlap):
        # max containment = 0.5 < 1.0, so blobs stay separate
        labels, n = label_temporal_neighbor(single_pixel_overlap, contain_thresh=1.0)
        assert n == 2, f"Expected 2 separate events, got {n}"
 
    def test_full_containment_merges(self, full_containment):
        # max containment = 1.0, threshold is exactly met — should merge
        labels, n = label_temporal_neighbor(full_containment, contain_thresh=1.0)
        assert n == 1, f"Expected 1 event (full containment), got {n}"
 
    def test_no_overlap_stays_separate(self, no_overlap):
        labels, n = label_temporal_neighbor(no_overlap, contain_thresh=1.0)
        assert n == 2, f"Expected 2 events, got {n}"
 
    def test_chain_breaks_into_independent_events(self, three_frames_chain):
        # 0.5 containment between adjacent frames < 1.0, so no linking
        labels, n = label_temporal_neighbor(three_frames_chain, contain_thresh=1.0)
        n_frames_with_blobs = int(
            (three_frames_chain.values.sum(axis=(1, 2)) > 0).sum()
        )
        assert n == n_frames_with_blobs, (
            f"Expected one event per frame ({n_frames_with_blobs}), got {n}"
        )
 
    def test_strictly_more_events_than_thresh0_on_partial_overlap(self, single_pixel_overlap):
        # raising thresh can only maintain or increase the event count
        _, n0 = label_temporal_neighbor(single_pixel_overlap, contain_thresh=0.0)
        _, n1 = label_temporal_neighbor(single_pixel_overlap, contain_thresh=1.0)
        assert n1 >= n0

# ── label_temporal_neighbor: threshold sensitivity ──────────────────────────────────
 
class TestContainThreshSweep:
    """
    Testing thresh from 0.0 to 1.0 and verify behaviour:
    event count is non-decreasing as thresh increases.
    """
 
    def test_monotone_increasing_event_count(self, single_pixel_overlap):
        thresholds = [0.0, 0.1, 0.25, 0.49, 0.50, 0.51, 0.75, 1.0]
        counts = [
            label_temporal_neighbor(single_pixel_overlap, contain_thresh=t)[1]
            for t in thresholds
        ]
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], (
                f"Event count decreased from thresh={thresholds[i]} "
                f"(n={counts[i]}) to thresh={thresholds[i+1]} (n={counts[i+1]})"
            )
 
    def test_boundary_at_exact_containment(self, single_pixel_overlap):
        # single_pixel_overlap has max containment = 0.5 exactly
        # thresh=0.50 satisfies >=, so should merge; 0.51 should not
        _, n_at   = label_temporal_neighbor(single_pixel_overlap, contain_thresh=0.50)
        _, n_just = label_temporal_neighbor(single_pixel_overlap, contain_thresh=0.51)
        assert n_at   == 1, f"thresh=0.50: expected 1 event, got {n_at}"
        assert n_just == 2, f"thresh=0.51: expected 2 events, got {n_just}"

# ── SurfTracker integration: contain_thresh via the full pipeline ─────────────
class TestSurfTrackerContainThresh:
    """
    Confirms contain_thresh propagates from constructor through track()
    and into result.attrs, and that method='3d' is unaffected by it.
    """
 
    def _blob_pair(self, overlap_cols=1):
        """
        Two-frame dataset. overlap_cols controls how many columns are shared.
            0  -> no overlap
            2  -> 50% containment
            4  -> identical frames, 100% containment
        """
        Y, X = 20, 40
        lat, lon = np.arange(Y, dtype=float), np.arange(X, dtype=float)
 
        f0 = np.zeros((Y, X)); f0[8:12, 5:9] = 1
        f1 = np.zeros((Y, X)); f1[8:12, 9 - overlap_cols:13 - overlap_cols] = 1
 
        da = xr.DataArray(
            np.stack([f0, f1]),
            dims=["time", "lat", "lon"],
            coords={"time": [0, 1], "lat": lat, "lon": lon},
        )
        mask = xr.DataArray(np.ones((Y, X)), dims=["lat", "lon"],
                            coords={"lat": lat, "lon": lon})
        return da, mask
 
    def _tracker(self, da, mask, method="temporal_neighbor", contain_thresh=0.0):
        return SurfTracker(
            da, mask,
            method=method, contain_thresh=contain_thresh,
            min_area_cells=1, min_size_quartile=0.0,
            xdim="lon", ydim="lat", radius=1,
        )
 
    def test_permissive_thresh_merges_more_than_strict(self):
        da, mask = self._blob_pair(overlap_cols=1)
        t0 = self._tracker(da, mask, contain_thresh=0.0)
        t1 = self._tracker(da, mask, contain_thresh=1.0)
        t0.run(); t1.run()
        assert t0.n_events() <= t1.n_events()
 
    def test_contain_thresh_written_to_attrs(self):
        da, mask = self._blob_pair(overlap_cols=2)
        for thresh in (0.0, 1.0):
            result = self._tracker(da, mask, contain_thresh=thresh).run()
            assert result.attrs["contain_thresh"] == thresh
 
    def test_no_overlap_always_two_events(self):
        da, mask = self._blob_pair(overlap_cols=0)
        for thresh in (0.0, 1.0):
            t = self._tracker(da, mask, contain_thresh=thresh)
            t.run()
            assert t.n_events() == 2
 
    def test_full_overlap_always_one_event(self):
        da, mask = self._blob_pair(overlap_cols=4)
        for thresh in (0.0, 1.0):
            t = self._tracker(da, mask, contain_thresh=thresh)
            t.run()
            assert t.n_events() == 1
 
    def test_method_3d_unaffected_by_contain_thresh(self):
        # contain_thresh is silently ignored for method='3d' — should not error
        da, mask = self._blob_pair(overlap_cols=2)
        for thresh in (0.0, 1.0):
            result = self._tracker(da, mask, method="3d", contain_thresh=thresh).run()
            assert isinstance(result, xr.DataArray)
            assert result.attrs["method"] == "3d"
 
    def test_both_methods_agree_on_single_persistent_blob(self):
        # identical frames: one blob that never splits — both methods should give 1 event
        da, mask = self._blob_pair(overlap_cols=4)
        t_3d = self._tracker(da, mask, method="3d",               contain_thresh=0.0)
        t_tn = self._tracker(da, mask, method="temporal_neighbor", contain_thresh=0.0)
        t_3d.run(); t_tn.run()
        assert t_3d.n_events() == t_tn.n_events() == 1
 
    def test_invalid_method_raises_on_init(self):
        da, mask = self._blob_pair()
        with pytest.raises(ValueError, match="method"):
            SurfTracker(da, mask, method="not_a_method", xdim="lon", ydim="lat")

# ── SurfTracker ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["3d", "temporal_neighbor"])
class TestSurfTrackerIntegration:
    """
    End-to-end pipeline tests run against both labelling methods.
    contain_thresh=0.0 throughout, testing the method, not the threshold.
    """

    def _tracker_data(self, T=4):
        lon = np.arange(0, 360) + 0.5
        lat = np.arange(-90, 90) + 0.5
        x, y = np.meshgrid(lon, lat)
        blob = np.exp(-((x - 180) ** 2 + y ** 2) / (2 * 20 ** 2))
        data = np.where(blob * 2 - 0.5 > 0, blob * 2 - 0.5, 0)
        data = np.stack([data] * T, axis=0)
        da   = xr.DataArray(data, dims=["time", "lat", "lon"],
                            coords={"time": np.arange(T), "lat": lat, "lon": lon})
        mask = xr.DataArray(np.ones((T, 180, 360)), dims=da.dims, coords=da.coords)
        return da, mask
 
    def _make_tracker(self, da, mask, method, **kwargs):
        return SurfTracker(
            da, mask,
            radius            = kwargs.get("radius", 5),
            min_size_quartile = kwargs.get("min_size_quartile", 0.5),
            min_area_cells    = kwargs.get("min_area_cells", 10),
            xdim              = "lon",
            ydim              = "lat",
            method            = method,
            contain_thresh    = 0.0,
        )
 
    def test_run_returns_dataarray(self, method):
        da, mask = self._tracker_data()
        result = self._make_tracker(da, mask, method).run()
        assert isinstance(result, xr.DataArray)
        assert result.shape == da.shape
 
    def test_result_background_is_nan(self, method):
        da, mask = self._tracker_data()
        result = self._make_tracker(da, mask, method).run()
        assert not (result.values == 0).any()
 
    def test_percent_area_sums_to_one(self, method):
        da, mask = self._tracker_data()
        result = self._make_tracker(da, mask, method).run()
        total = (result.attrs["percent area reject"]
                 + result.attrs["percent area accept"])
        assert total == pytest.approx(1.0)
 
    def test_method_stored_in_attrs(self, method):
        """result.attrs['method'] must match the method used."""
        da, mask = self._tracker_data()
        result = self._make_tracker(da, mask, method).run()
        assert result.attrs["method"] == method
 
    def test_n_events_positive(self, method):
        da, mask = self._tracker_data()
        t = self._make_tracker(da, mask, method)
        t.run()
        assert t.n_events() >= 1
 
    def test_event_duration_dict(self, method):
        da, mask = self._tracker_data(T=5)
        t = self._make_tracker(da, mask, method)
        t.run()
        durations = t.event_duration()
        assert isinstance(durations, dict)
        for eid, dur in durations.items():
            assert isinstance(eid, int) and dur > 0
 
    def test_summary_runs(self, method):
        da, mask = self._tracker_data()
        t = self._make_tracker(da, mask, method)
        t.run()
        t.summary()   # should not raise
 
    def test_repr_before_run(self, method):
        da, mask = self._tracker_data()
        assert "(not run yet)" in repr(
            SurfTracker(da, mask, xdim="lon", ydim="lat", method=method)
        )
 
    def test_postprocess_raises_without_track(self, method):
        da, mask = self._tracker_data()
        with pytest.raises(RuntimeError, match="track"):
            SurfTracker(da, mask, xdim="lon", ydim="lat", method=method).postprocess()
 
    def test_chaining_returns_self(self, method):
        da, mask = self._tracker_data()
        t = self._make_tracker(da, mask, method)
        assert t.clean()  is t
        assert t.filter() is t
        assert t.track()  is t
 
    def test_step_by_step_matches_run(self, method):
        """Running steps individually must produce the same result as .run()."""
        da, mask = self._tracker_data()
 
        r1 = self._make_tracker(da, mask, method).run()
 
        t2 = self._make_tracker(da, mask, method)
        t2.clean().filter().track().postprocess()
        r2 = t2.result
 
        np.testing.assert_array_equal(
            np.nan_to_num(r1.values),
            np.nan_to_num(r2.values),
        )
 
    def test_min_area_cells_removes_small_events(self, method):
        """Change 1 — high min_area_cells removes small objects the percentile would keep."""
        da, mask = self._tracker_data()
 
        t_high = self._make_tracker(da, mask, method,
                                    min_size_quartile=0.0, min_area_cells=999999)
        t_low  = self._make_tracker(da, mask, method,
                                    min_size_quartile=0.0, min_area_cells=1)
        result_high = t_high.run()
        result_low  = t_low.run()
 
        n_high = len(np.unique(result_high.values[~np.isnan(result_high.values)]))
        n_low  = len(np.unique(result_low.values[~np.isnan(result_low.values)]))
        assert n_high <= n_low
 
    def test_chunked_input_works(self, method):
        """Change 3 — spatially chunked Dask input must not raise in morphological_operations."""
        da, mask = self._tracker_data()
        da_chunked = da.chunk({"time": 1, "lat": 45, "lon": 90})
        result = self._make_tracker(da_chunked, mask, method).run()
        assert result is not None
        