# ============================================================
# tests/test_deeptrack.py
# ============================================================
"""
Tests for ocetrac/DeepTrack.

Run with:  pytest tests/test_deeptrack.py -v

The tests import from DeepTrack (add ocetrac/ to sys.path first
if running outside the package):

    cd ocetrac/
    pytest ../tests/test_deeptrack.py -v
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import xarray as xr


# helpers --------------------

def _da(data: np.ndarray) -> xr.DataArray:
    """Wrap ndarray as a DataArray with standard DeepTrack dims."""
    dims = ["time", "z_t", "nlat", "nlon"]
    coords = {
        "time": np.arange(data.shape[0]),
        "z_t":  np.arange(data.shape[1]) * 500.0,
        "nlat": np.arange(data.shape[2]),
        "nlon": np.arange(data.shape[3]),
    }
    return xr.DataArray(data, dims=dims, coords=coords)


def _tracker_da(T=4, Z=3, Y=12, X=12, val=2.0) -> xr.DataArray:
    """Small DataArray with one event for integration tests."""
    data = np.zeros((T, Z, Y, X))
    data[:, :, 4:8, 4:8] = val
    return _da(data)


# grid --------------------
# all POP-specific or POP-assumed/for POP-style grid

class TestComputeDz:
    
    def test_shape(self):
        # output array should have the same length as the input depth vector
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([0., 10., 30., 70.]))
        assert dz.shape == (4,)

    def test_central_difference(self):
        # interior points use central differencing: dz[1] = (z[2] - z[0]) / 2
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([0., 10., 30., 70.]))
        assert np.isclose(dz[1], 15.0)   # (30-0)/2

    def test_boundary_differences(self):
        # first point uses forward difference, last uses backward difference
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([0., 5., 15.]))
        assert np.isclose(dz[0],  5.0)   # forward
        assert np.isclose(dz[-1], 10.0)  # backward

    def test_uniform_grid(self):
        # first point uses forward difference, last uses backward difference
        from ocetrac.DeepTrack.grid import compute_dz
        assert np.allclose(compute_dz(np.arange(5) * 10.0), 10.0)

    def test_single_level_returns_zero(self):
        # a single depth level has no neighbours so dz is undefined — return 0
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([50.0]))
        assert dz.shape == (1,)
        assert dz[0] == 0.0


class TestBuildCellVolume:
    
    def test_shape(self):
        # output must be (n_z, nlat, nlon)
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((4, 5)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500., 2500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=3)
        assert cv.shape == (3, 4, 5)

    def test_dims(self):
        # dimension order must be (z_t, nlat, nlon) for broadcasting in tracking
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((4, 5)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500., 2500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=3)
        assert tuple(cv.dims) == ("z_t", "nlat", "nlon")

    def test_positive_values(self):
        # every cell volume must be strictly positive, negative or zero
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((3, 3)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=2)
        assert (cv.values > 0).all()

    def test_units_m3(self):
        # POP inputs: TAREA in cm², z_t in cm → output must be in m³
        # 1e8 cm² × 1000 cm dz → 1e11 cm³ → 1e5 m³
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((1, 1)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=2)
        # dz[0] = 1500-500 = 1000 cm = 10 m; area = 1e8 cm² = 1e4 m²
        assert np.isclose(cv.values[0, 0, 0], 1e4 * 10, rtol=0.01)


class TestMakeAnisotropicStruct:
    def test_shape(self):
        # structuring element must be (3, 3, 3) for scipy.ndimage.label
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        assert make_anisotropic_struct().shape == (3, 3, 3)

    def test_centre_always_true(self):
        # the centre voxel must always be True regardless of connectivity flags
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        for xy in (True, False):
            for z in (True, False):
                assert make_anisotropic_struct(xy, z)[1, 1, 1]

    def test_no_connectivity(self):
        # with both flags off only the centre is True so no neighbours connected
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        s = make_anisotropic_struct(connect_xy=False, connect_z=False)
        assert s.sum() == 1

    def test_z_connectivity_only(self):
        # with only z connectivity, the two vertical face neighbours are True
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        s = make_anisotropic_struct(connect_xy=False, connect_z=True)
        assert s[0, 1, 1] and s[2, 1, 1]
        assert s.sum() == 3

    def test_xy_connectivity_only(self):
        # with only xy connectivity, the middle z-plane is fully connected
        # (8-connectivity + centre = 9) and the z-planes above/below are empty
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        s = make_anisotropic_struct(connect_xy=True, connect_z=False)
        assert s[0].sum() == 0 and s[2].sum() == 0
        assert s[1].sum() == 9   # full 8-connectivity + centre


# 2D labelling --------------------

class TestLabel2dStack:
    def test_shape(self):
        # output shape must match input shape exactly
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((3, 2, 8, 8), dtype=bool)
        binary[:, :, 1:3, 1:3] = True
        out = label_2d_stack(_da(binary))
        assert out.shape == (3, 2, 8, 8)

    def test_background_is_zero(self):
        # cells outside any blob must be labelled 0
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((2, 2, 6, 6), dtype=bool)
        binary[:, :, 2:4, 2:4] = True
        out = label_2d_stack(_da(binary))
        assert out.values[0, 0, 0, 0] == 0

    def test_two_separate_blobs_get_different_ids(self):
        # two disconnected blobs in the same (t, z) slice get different IDs
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((1, 1, 8, 8), dtype=bool)
        binary[0, 0, 0:2, 0:2] = True
        binary[0, 0, 5:7, 5:7] = True
        out = label_2d_stack(_da(binary))
        ids = np.unique(out.values)
        assert len(ids[ids > 0]) == 2

    def test_one_blob_one_id(self):
        # a single connected blob must receive exactly one unique label
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((2, 2, 6, 6), dtype=bool)
        binary[:, :, 2:4, 2:4] = True
        out = label_2d_stack(_da(binary))
        ids = np.unique(out.values)
        assert len(ids[ids > 0]) == 1


# area filter --------------------

class TestFilterArea2dGlobalDepth:
    def test_shape_preserved(self):
        # filtering must not change the array shape
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((3, 2, 10, 10), dtype=int)
        assert filter_area_2d_global_depth(arr).shape == arr.shape

    def test_empty_stays_zero(self):
        # an all-zero input has no blobs to filter — output must be all zeros
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((4, 2, 8, 8), dtype=int)
        assert (filter_area_2d_global_depth(arr) == 0).all()

    def test_removes_blob_below_absolute_floor(self):
        # blobs smaller than min_area_cells are zeroed out
        # blobs larger than the floor survive
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((4, 1, 20, 20), dtype=int)
        arr[:, 0, 0:2, 0:2] = 1       # 4 cells — below floor of 10
        arr[:, 0, 5:15, 5:15] = 2     # 100 cells — above floor
        out = filter_area_2d_global_depth(arr, min_area_cells=10)
        assert (out[:, 0, 0:2, 0:2] == 0).all()
        assert (out[:, 0, 5:15, 5:15] > 0).all()

    def test_keeps_large_blob(self):
        # a blob well above the area floor should not be removed
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((2, 1, 20, 20), dtype=int)
        arr[:, 0, 5:15, 5:15] = 1
        out = filter_area_2d_global_depth(arr, min_area_cells=10)
        assert (out[:, 0, 5:15, 5:15] > 0).all()

    def test_absolute_floor_beats_low_percentile(self):
        # even with min_quantile=0 (no percentile filtering), min_area_cells
        # must still remove blobs below the absolute floor
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((2, 1, 20, 20), dtype=int)
        arr[0, 0, 0:2, 0:2] = 1    # 4 cells
        out = filter_area_2d_global_depth(arr, min_quantile=0.0, min_area_cells=10)
        assert (out == 0).all()


class TestRelabel2d:
    def test_relabels_consecutively(self):
        # non-consecutive IDs (e.g. 5, 9) must be remapped to 1, 2
        from ocetrac.DeepTrack.tracker import relabel_2d
        arr = np.zeros((6, 6), dtype=int)
        arr[0:2, 0:2] = 5
        arr[4:6, 4:6] = 9
        out = relabel_2d(arr)
        assert sorted(np.unique(out).tolist()) == [0, 1, 2]

    def test_empty_stays_zero(self):
        # an all-zero slice has nothing to relabel — output must be all zeros
        from ocetrac.DeepTrack.tracker import relabel_2d
        assert (relabel_2d(np.zeros((4, 4), dtype=int)) == 0).all()


# 3d connectivity --------------------

class TestBuild3dObjects:
    def test_shape(self):
        # output shape must match input shape
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((3, 4, 8, 8), dtype=int)
        arr[:, :, 2:4, 2:4] = 1
        out = build_3d_objects(arr, make_anisotropic_struct())
        assert out.shape == (3, 4, 8, 8)

    def test_single_blob_single_label(self):
        # one connected 3-D blob must receive exactly one label
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((2, 3, 6, 6), dtype=int)
        arr[0, :, 2:4, 2:4] = 1
        out = build_3d_objects(arr, make_anisotropic_struct())
        ids = np.unique(out[0])
        assert len(ids[ids > 0]) == 1

    def test_labels_reset_per_timestep(self):
        # labels are assigned independently at each timestep — both t=0 and
        # t=1 can have label 1 without being the same physical object
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((2, 2, 6, 6), dtype=int)
        arr[0, :, 0:2, 0:2] = 1   # blob at t=0
        arr[1, :, 4:6, 4:6] = 1   # different blob at t=1
        out = build_3d_objects(arr, make_anisotropic_struct())
        # Both get label 1 — they are independent
        assert out[0, 0, 0, 0] == 1
        assert out[1, 0, 5, 5] == 1

    def test_two_disconnected_blobs_get_different_labels(self):
        # two spatially separate blobs at the same timestep must get different IDs
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((1, 2, 8, 8), dtype=int)
        arr[0, :, 0:2, 0:2] = 1
        arr[0, :, 5:7, 5:7] = 1
        out = build_3d_objects(arr, make_anisotropic_struct())
        ids = np.unique(out[0])
        assert len(ids[ids > 0]) == 2


# volume pre-filter --------------------

class TestFilterPreserveLabelsGlobal:
    def test_removes_smallest_fraction(self):
        # the bottom frac fraction of objects by total voxel count must be removed
        # original label IDs of surviving objects must be preserved
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        tracks = np.zeros((4, 3, 8, 8), dtype=int)
        tracks[:, :, 2:6, 2:6] = 1    # large
        tracks[0, 0, 0, 0]     = 2    # tiny (1 voxel)
        out = filter_preserve_labels_global(tracks, frac=0.5)
        assert (out == 2).sum() == 0
        assert (out == 1).sum() > 0

    def test_empty_returns_zeros(self):
        # an all-zero input has no objects to filter so output must be all zeros
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        out = filter_preserve_labels_global(np.zeros((2, 2, 4, 4), dtype=int))
        assert (out == 0).all()

    def test_preserves_original_ids(self):
        # surviving objects must keep their original label IDs unchanged
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        tracks = np.zeros((3, 2, 6, 6), dtype=int)
        tracks[:, :, 1:4, 1:4] = 7
        out = filter_preserve_labels_global(tracks, frac=0.0)
        assert np.all((out == 0) | (out == 7))

    def test_frac_zero_keeps_everything(self):
        # frac=0 means nothing is removed
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        tracks = np.zeros((2, 2, 6, 6), dtype=int)
        tracks[:, :, 0:2, 0:2] = 1
        tracks[:, :, 3:5, 3:5] = 2
        out = filter_preserve_labels_global(tracks, frac=0.0)
        assert (out == 1).any() and (out == 2).any()

# tracker --------------------

class TestTrackObjectsWithSplitting:
    def _in(self, data):
        return xr.DataArray(
            data, dims=["time", "z_t", "nlat", "nlon"],
            coords={k: np.arange(s) for k, s in
                    zip(["time", "z_t", "nlat", "nlon"], data.shape)},
        )

    def test_return_types(self):
        # must return (ndarray, dict) — not a DataArray, not three values
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((3, 1, 6, 6), dtype=int)
        tracked, origin_map = track_objects_with_splitting(data)
        assert isinstance(tracked, np.ndarray)
        assert isinstance(origin_map, dict)

    def test_output_shape(self):
        # output array must have the same shape as the input
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((4, 1, 6, 6), dtype=int)
        tracked, _ = track_objects_with_splitting(data)
        assert tracked.shape == (4, 1, 6, 6)

    def test_empty_input_all_zeros(self):
        # no objects to track
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((3, 1, 6, 6), dtype=int)
        tracked, origin_map = track_objects_with_splitting(data)
        assert (tracked == 0).all()
        assert len(origin_map) == 0

    def test_persistent_event_keeps_same_id(self):
        # a blob that persists with full overlap across all timesteps must
        # receive the same event ID at every timestep
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((4, 1, 6, 6), dtype=int)
        data[:, :, 2:4, 2:4] = 1
        tracked, _ = track_objects_with_splitting(
            self._in(data), contain_thresh=0.1
        )
        ids = [set(np.unique(tracked[t, 0, 2:4, 2:4])) - {0} for t in range(4)]
        assert ids[0] == ids[1] == ids[2] == ids[3]

    def test_unrelated_event_gets_new_id(self):
        # a blob at t=1 with no spatial overlap with t=0 must get a new event ID
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((2, 1, 8, 8), dtype=int)
        data[0, 0, 0:2, 0:2] = 1   # top-left
        data[1, 0, 6:8, 6:8] = 1   # bottom-right — no overlap
        tracked, _ = track_objects_with_splitting(
            self._in(data), contain_thresh=0.3
        )
        assert int(tracked[0, 0, 1, 1]) != int(tracked[1, 0, 7, 7])

    def test_split_children_inherit_parent_id(self):
        # when a large blob splits into two children, both children must
        # inherit the parent's event ID (lineage preservation)
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((2, 1, 8, 8), dtype=int)
        data[0, 0, 1:6, 1:6] = 1   # large parent
        data[1, 0, 1:3, 1:3] = 1   # child A — inside parent
        data[1, 0, 4:6, 4:6] = 2   # child B — inside parent
        tracked, _ = track_objects_with_splitting(
            self._in(data), contain_thresh=0.05
        )
        parent_id = int(tracked[0, 0, 2, 2])
        assert int(tracked[1, 0, 2, 2]) == parent_id
        assert int(tracked[1, 0, 5, 5]) == parent_id

    def test_origin_map_self_for_clean_events(self):
        # origin_map maps every event ID to its previous
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((3, 1, 6, 6), dtype=int)
        data[:, :, 2:4, 2:4] = 1
        _, origin_map = track_objects_with_splitting(
            self._in(data), contain_thresh=0.1
        )
        for eid, orig in origin_map.items():
            assert isinstance(orig, int)

    def test_returns_two_values_not_three(self):
        # regression: old type hint said 3 returns, actual signature returns 2
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((2, 1, 4, 4), dtype=int)
        result = track_objects_with_splitting(data)
        assert len(result) == 2


# utils --------------------

class TestComputeDaskQuantile:
    def test_shape_removes_time(self):
        # quantile is computed over the time axis
        from ocetrac.DeepTrack.utils import compute_dask_quantile
        da = _da(np.random.rand(10, 3, 4, 5)).chunk({"time": 5})
        q  = compute_dask_quantile(da, q=0.9)
        assert q.shape == (3, 4, 5)
        assert "time" not in q.dims

    def test_median_value(self):
        # median of 0..9 is 4.5
        from ocetrac.DeepTrack.utils import compute_dask_quantile
        data = np.arange(10, dtype=float).reshape(10, 1, 1, 1)
        da   = _da(data).chunk({"time": 5})
        q    = compute_dask_quantile(da, q=0.5).compute().values.item()
        assert abs(q - 4.5) < 0.5

# TestDeepTrackerIntegration --------------------

class TestDeepTrackerIntegration:
    
    def _tracker(self, T=5, Z=3, Y=12, X=12):
        da = _tracker_da(T=T, Z=Z, Y=Y, X=X)
        cv = np.ones((Z, Y, X)) * 1e6
        return da, cv

    def test_run_returns_dataarray(self):
        # full pipeline must return an xr.DataArray with the same shape as input
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        result = DeepTracker(da, radius=1, min_area_cells=1,
                             frac_filter=0.).run(cell_volume=cv)
        assert isinstance(result, xr.DataArray)
        assert result.shape == da.shape

    def test_result_background_is_nan(self):
        # background cells (originally 0) must be NaN in the result —
        # zeros should never appear since they'd be confused with background
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        arr = t.result.values
        # Background cells (originally 0) must be NaN in result
        assert not (arr == 0).any()

    def test_n_events_positive(self):
        # at least one event must be detected in a dataset with a clear blob
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        assert t.n_events() >= 1

    def test_event_duration_dict(self):
        # event_duration must return a dict mapping int IDs to positive int durations
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker(T=5)
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        durations = t.event_duration()
        assert isinstance(durations, dict)
        for eid, dur in durations.items():
            assert isinstance(eid, int) and isinstance(dur, int) and dur > 0

    def test_summary_runs_without_error(self):
        # summary() prints diagnostics
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        t.summary()

    def test_repr_before_run(self):
        # before run(), repr signals that results are not yet available
        from ocetrac.DeepTrack import DeepTracker
        da, _ = self._tracker()
        assert "(not run yet)" in repr(DeepTracker(da))

    def test_repr_after_run(self):
        # after run(), repr must include the class name and drop the not-run message
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        assert "DeepTracker(" in repr(t)
        assert "(not run yet)" not in repr(t)

    def test_postprocess_raises_without_track(self):
        # calling postprocess() before track() must raise RuntimeError
        from ocetrac.DeepTrack import DeepTracker
        da, _ = self._tracker()
        with pytest.raises(RuntimeError, match="track"):
            DeepTracker(da).postprocess()

    def test_n_events_raises_before_run(self):
        # calling n_events() before run() must raise RuntimeError
        from ocetrac.DeepTrack import DeepTracker
        da, _ = self._tracker()
        with pytest.raises(RuntimeError):
            DeepTracker(da).n_events()

    def test_chaining_returns_self(self):
        # each pipeline method must return self to support method chaining
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        assert t.clean() is t
        assert t.label() is t
        assert t.connect_depth() is t
        assert t.prefilter() is t
        assert t.track(cell_volume=cv) is t

    def test_step_by_step_matches_run(self):
        # running each step individually must produce the same result as run()
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()

        t1 = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        r1 = t1.run(cell_volume=cv)

        t2 = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t2.clean().label().connect_depth().prefilter().track(cell_volume=cv).postprocess()
        r2 = t2.result

        np.testing.assert_array_equal(
            np.nan_to_num(r1.values),
            np.nan_to_num(r2.values),
        )

# testing wrap longitude --------------------
### Not sure if I'm missing other test cases

class TestWrapLongitude:
 
    def _arr(self, T=2, Z=2, Y=6, X=10):
        return np.zeros((T, Z, Y, X), dtype=int)
 
    def test_straddling_blob_merged_smallest_id_wins(self):
        # two labels on opposite edges at the same (t, z, y) 
        # position are the same physical object so they
        # should merge into the smaller object
        from ocetrac.DeepTrack import _wrap_longitude
        arr = self._arr()
        arr[0, 0, 3,  0] = 7   # left edge
        arr[0, 0, 3, -1] = 3   # right edge, same (t, z, y)
        out = _wrap_longitude(arr)
        assert out[0, 0, 3, 0] == 3
        assert out[0, 0, 3, -1] == 3
 
    def test_different_y_and_interior_not_merged(self):
        # labels at different y positions on opposite edges
        # are separate objects
        # interior object nowhere near the edges should not be
        # touched
        from ocetrac.DeepTrack import _wrap_longitude
        arr = self._arr()
        arr[0, 0, 1,  0] = 1   # left edge at y=1
        arr[0, 0, 4, -1] = 2   # right edge at y=4 — different row
        arr[0, 0, 1, 4:6] = 3  # interior blob
        out = _wrap_longitude(arr)
        assert out[0, 0, 1, 0] == 1    # not merged
        assert out[0, 0, 4, -1] == 2   # not merged
        assert (out[0, 0, 1, 4:6] == 3).all()  # interior untouched

    def test_n_labels_nonincreasing(self):
        # wrapping can only merge labels, not create new ones
        # number of unique labels after wrapping must be <=
        # the number before
        from ocetrac.DeepTrack import _wrap_longitude
        arr = self._arr()
        arr[0, 0, 2,  0] = 1;  arr[0, 0, 2, -1] = 2
        arr[1, 1, 4,  0] = 3;  arr[1, 1, 4, -1] = 4
        n_before = len(np.unique(arr[arr > 0]))
        out      = _wrap_longitude(arr)
        n_after  = len(np.unique(out[out > 0]))
        assert n_after <= n_before

class TestDeepTrackerWrapLon:
 
    def _global_da(self, T=3, Z=2, Y=10, X=20):
        # one object split at the date line (left and right edges, 
        # same rows, and an separate interior object
        data = np.zeros((T, Z, Y, X))
        data[:, :, 4:7, 8:12] = 2.0   # interior
        data[:, :, 1:4,  :2 ] = 2.0   # left edge  
        data[:, :, 1:4, -2: ] = 2.0   # right edge 
        lon = np.linspace(0, 360, X, endpoint=False)
        lat = np.linspace(-45, 45, Y)
        return xr.DataArray(
            data,
            dims=["time", "z_t", "nlat", "nlon"],
            coords={"time": np.arange(T), "z_t": np.arange(Z) * 500.,
                    "nlat": lat, "nlon": lon},
        )
 
    def test_default_is_false_and_stored_in_repr(self):
        # wrap_lon should default to False so regional domain
        # instances are unaffected
        from ocetrac.DeepTrack import DeepTracker
        da = self._global_da()
        assert DeepTracker(da).wrap_lon is False
        assert "wrap_lon=True"  in repr(DeepTracker(da, wrap_lon=True))
        assert "wrap_lon=False" in repr(DeepTracker(da, wrap_lon=False))
 
    def test_wrap_true_merges_split_wrap_false_does_not(self):
        # wrap_lon=True: left and right edge cells of the date-line object
        # must share one label ID at every timestep after connect_depth
        # wrap_lon=True must never produce more labels than wrap_lon=False
        from ocetrac.DeepTrack import DeepTracker
        da = self._global_da()
 
        t_yes = DeepTracker(da, radius=1, min_area_cells=1,
                            frac_filter=0., wrap_lon=True)
        t_no  = DeepTracker(da, radius=1, min_area_cells=1,
                            frac_filter=0., wrap_lon=False)
        t_yes.clean().label().connect_depth()
        t_no.clean().label().connect_depth()
 
        # wrap=True: left and right edge cells share one ID at every timestep
        for ts in range(t_yes.labeled_3d.shape[0]):
            left  = set(np.unique(t_yes.labeled_3d[ts, :, 1:4,  :2])) - {0}
            right = set(np.unique(t_yes.labeled_3d[ts, :, 1:4, -2:])) - {0}
            if left and right:
                assert left == right
 
        # wrap=True never creates more labels than wrap=False
        n_yes = len(np.unique(t_yes.labeled_3d[t_yes.labeled_3d > 0]))
        n_no  = len(np.unique(t_no.labeled_3d[ t_no.labeled_3d  > 0]))
        assert n_yes <= n_no