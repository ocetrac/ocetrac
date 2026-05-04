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


# ── helpers
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
    """Small DataArray with one warm blob for integration tests."""
    data = np.zeros((T, Z, Y, X))
    data[:, :, 4:8, 4:8] = val
    return _da(data)


# ============================================================
# grid
# ============================================================

class TestComputeDz:
    def test_shape(self):
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([0., 10., 30., 70.]))
        assert dz.shape == (4,)

    def test_central_difference(self):
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([0., 10., 30., 70.]))
        assert np.isclose(dz[1], 15.0)   # (30-0)/2

    def test_boundary_differences(self):
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([0., 5., 15.]))
        assert np.isclose(dz[0],  5.0)   # forward
        assert np.isclose(dz[-1], 10.0)  # backward

    def test_uniform_grid(self):
        from ocetrac.DeepTrack.grid import compute_dz
        assert np.allclose(compute_dz(np.arange(5) * 10.0), 10.0)

    def test_single_level_returns_zero(self):
        from ocetrac.DeepTrack.grid import compute_dz
        dz = compute_dz(np.array([50.0]))
        assert dz.shape == (1,)
        assert dz[0] == 0.0


class TestBuildCellVolume:
    def test_shape(self):
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((4, 5)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500., 2500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=3)
        assert cv.shape == (3, 4, 5)

    def test_dims(self):
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((4, 5)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500., 2500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=3)
        assert tuple(cv.dims) == ("z_t", "nlat", "nlon")

    def test_positive_values(self):
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((3, 3)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=2)
        assert (cv.values > 0).all()

    def test_units_m3(self):
        """1e8 cm² × 1000 cm → 1e11 cm³ → 1e5 m³ for the 1000 cm layer."""
        from ocetrac.DeepTrack.grid import build_cell_volume
        TAREA = xr.DataArray(np.ones((1, 1)) * 1e8, dims=["nlat", "nlon"])
        z_t   = xr.DataArray(np.array([500., 1500.]), dims=["z_t"])
        cv    = build_cell_volume(TAREA, z_t, n_z=2)
        # dz[0] = 1500-500 = 1000 cm = 10 m; area = 1e8 cm² = 1e4 m²
        assert np.isclose(cv.values[0, 0, 0], 1e4 * 10, rtol=0.01)


class TestMakeAnisotropicStruct:
    def test_shape(self):
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        assert make_anisotropic_struct().shape == (3, 3, 3)

    def test_centre_always_true(self):
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        for xy in (True, False):
            for z in (True, False):
                assert make_anisotropic_struct(xy, z)[1, 1, 1]

    def test_no_connectivity(self):
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        s = make_anisotropic_struct(connect_xy=False, connect_z=False)
        assert s.sum() == 1

    def test_z_connectivity_only(self):
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        s = make_anisotropic_struct(connect_xy=False, connect_z=True)
        assert s[0, 1, 1] and s[2, 1, 1]
        assert s.sum() == 3

    def test_xy_connectivity_only(self):
        from ocetrac.DeepTrack.grid import make_anisotropic_struct
        s = make_anisotropic_struct(connect_xy=True, connect_z=False)
        assert s[0].sum() == 0 and s[2].sum() == 0
        assert s[1].sum() == 9   # full 8-connectivity + centre


# ============================================================
# tracker — 2-D labelling
# ============================================================

class TestLabel2dStack:
    def test_shape(self):
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((3, 2, 8, 8), dtype=bool)
        binary[:, :, 1:3, 1:3] = True
        out = label_2d_stack(_da(binary))
        assert out.shape == (3, 2, 8, 8)

    def test_background_is_zero(self):
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((2, 2, 6, 6), dtype=bool)
        binary[:, :, 2:4, 2:4] = True
        out = label_2d_stack(_da(binary))
        assert out.values[0, 0, 0, 0] == 0

    def test_two_separate_blobs_get_different_ids(self):
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((1, 1, 8, 8), dtype=bool)
        binary[0, 0, 0:2, 0:2] = True
        binary[0, 0, 5:7, 5:7] = True
        out = label_2d_stack(_da(binary))
        ids = np.unique(out.values)
        assert len(ids[ids > 0]) == 2

    def test_one_blob_one_id(self):
        from ocetrac.DeepTrack.tracker import label_2d_stack
        binary = np.zeros((2, 2, 6, 6), dtype=bool)
        binary[:, :, 2:4, 2:4] = True
        out = label_2d_stack(_da(binary))
        ids = np.unique(out.values)
        assert len(ids[ids > 0]) == 1


# ============================================================
# tracker — area filter
# ============================================================

class TestFilterArea2dGlobalDepth:
    def test_shape_preserved(self):
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((3, 2, 10, 10), dtype=int)
        assert filter_area_2d_global_depth(arr).shape == arr.shape

    def test_empty_stays_zero(self):
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((4, 2, 8, 8), dtype=int)
        assert (filter_area_2d_global_depth(arr) == 0).all()

    def test_removes_blob_below_absolute_floor(self):
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((4, 1, 20, 20), dtype=int)
        arr[:, 0, 0:2, 0:2] = 1       # 4 cells — below floor of 10
        arr[:, 0, 5:15, 5:15] = 2     # 100 cells — above floor
        out = filter_area_2d_global_depth(arr, min_area_cells=10)
        assert (out[:, 0, 0:2, 0:2] == 0).all()
        assert (out[:, 0, 5:15, 5:15] > 0).all()

    def test_keeps_large_blob(self):
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((2, 1, 20, 20), dtype=int)
        arr[:, 0, 5:15, 5:15] = 1
        out = filter_area_2d_global_depth(arr, min_area_cells=10)
        assert (out[:, 0, 5:15, 5:15] > 0).all()

    def test_absolute_floor_beats_low_percentile(self):
        """Even if percentile threshold is 0, min_area_cells still removes tiny blobs."""
        from ocetrac.DeepTrack.tracker import filter_area_2d_global_depth
        arr = np.zeros((2, 1, 20, 20), dtype=int)
        arr[0, 0, 0:2, 0:2] = 1    # 4 cells
        out = filter_area_2d_global_depth(arr, min_quantile=0.0, min_area_cells=10)
        assert (out == 0).all()


class TestRelabel2d:
    def test_relabels_consecutively(self):
        from ocetrac.DeepTrack.tracker import relabel_2d
        arr = np.zeros((6, 6), dtype=int)
        arr[0:2, 0:2] = 5
        arr[4:6, 4:6] = 9
        out = relabel_2d(arr)
        assert sorted(np.unique(out).tolist()) == [0, 1, 2]

    def test_empty_stays_zero(self):
        from ocetrac.DeepTrack.tracker import relabel_2d
        assert (relabel_2d(np.zeros((4, 4), dtype=int)) == 0).all()


# ============================================================
# tracker — 3-D connectivity
# ============================================================

class TestBuild3dObjects:
    def test_shape(self):
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((3, 4, 8, 8), dtype=int)
        arr[:, :, 2:4, 2:4] = 1
        out = build_3d_objects(arr, make_anisotropic_struct())
        assert out.shape == (3, 4, 8, 8)

    def test_single_blob_single_label(self):
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((2, 3, 6, 6), dtype=int)
        arr[0, :, 2:4, 2:4] = 1
        out = build_3d_objects(arr, make_anisotropic_struct())
        ids = np.unique(out[0])
        assert len(ids[ids > 0]) == 1

    def test_labels_reset_per_timestep(self):
        """Labels at t=0 and t=1 are independent — both can have label 1."""
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
        from ocetrac.DeepTrack.tracker import build_3d_objects
        from ocetrac.DeepTrack.grid   import make_anisotropic_struct
        arr = np.zeros((1, 2, 8, 8), dtype=int)
        arr[0, :, 0:2, 0:2] = 1
        arr[0, :, 5:7, 5:7] = 1
        out = build_3d_objects(arr, make_anisotropic_struct())
        ids = np.unique(out[0])
        assert len(ids[ids > 0]) == 2


# ============================================================
# tracker — volume prefilter
# ============================================================

class TestFilterPreserveLabelsGlobal:
    def test_removes_smallest_fraction(self):
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        tracks = np.zeros((4, 3, 8, 8), dtype=int)
        tracks[:, :, 2:6, 2:6] = 1    # large
        tracks[0, 0, 0, 0]     = 2    # tiny (1 voxel)
        out = filter_preserve_labels_global(tracks, frac=0.5)
        assert (out == 2).sum() == 0
        assert (out == 1).sum() > 0

    def test_empty_returns_zeros(self):
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        out = filter_preserve_labels_global(np.zeros((2, 2, 4, 4), dtype=int))
        assert (out == 0).all()

    def test_preserves_original_ids(self):
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        tracks = np.zeros((3, 2, 6, 6), dtype=int)
        tracks[:, :, 1:4, 1:4] = 7
        out = filter_preserve_labels_global(tracks, frac=0.0)
        assert np.all((out == 0) | (out == 7))

    def test_frac_zero_keeps_everything(self):
        from ocetrac.DeepTrack.tracker import filter_preserve_labels_global
        tracks = np.zeros((2, 2, 6, 6), dtype=int)
        tracks[:, :, 0:2, 0:2] = 1
        tracks[:, :, 3:5, 3:5] = 2
        out = filter_preserve_labels_global(tracks, frac=0.0)
        assert (out == 1).any() and (out == 2).any()


# ============================================================
# tracker — track_objects_with_splitting
# ============================================================

class TestTrackObjectsWithSplitting:
    def _in(self, data):
        return xr.DataArray(
            data, dims=["time", "z_t", "nlat", "nlon"],
            coords={k: np.arange(s) for k, s in
                    zip(["time", "z_t", "nlat", "nlon"], data.shape)},
        )

    def test_return_types(self):
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((3, 1, 6, 6), dtype=int)
        tracked, origin_map = track_objects_with_splitting(data)
        assert isinstance(tracked, np.ndarray)
        assert isinstance(origin_map, dict)

    def test_output_shape(self):
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((4, 1, 6, 6), dtype=int)
        tracked, _ = track_objects_with_splitting(data)
        assert tracked.shape == (4, 1, 6, 6)

    def test_empty_input_all_zeros(self):
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((3, 1, 6, 6), dtype=int)
        tracked, origin_map = track_objects_with_splitting(data)
        assert (tracked == 0).all()
        assert len(origin_map) == 0

    def test_persistent_event_keeps_same_id(self):
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((4, 1, 6, 6), dtype=int)
        data[:, :, 2:4, 2:4] = 1
        tracked, _ = track_objects_with_splitting(
            self._in(data), contain_thresh=0.1
        )
        ids = [set(np.unique(tracked[t, 0, 2:4, 2:4])) - {0} for t in range(4)]
        assert ids[0] == ids[1] == ids[2] == ids[3]

    def test_unrelated_event_gets_new_id(self):
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((2, 1, 8, 8), dtype=int)
        data[0, 0, 0:2, 0:2] = 1   # top-left
        data[1, 0, 6:8, 6:8] = 1   # bottom-right — no overlap
        tracked, _ = track_objects_with_splitting(
            self._in(data), contain_thresh=0.3
        )
        assert int(tracked[0, 0, 1, 1]) != int(tracked[1, 0, 7, 7])

    def test_split_children_inherit_parent_id(self):
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
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((3, 1, 6, 6), dtype=int)
        data[:, :, 2:4, 2:4] = 1
        _, origin_map = track_objects_with_splitting(
            self._in(data), contain_thresh=0.1
        )
        for eid, orig in origin_map.items():
            assert isinstance(orig, int)

    def test_returns_two_values_not_three(self):
        """Regression: old type hint said 3 returns, actual is 2."""
        from ocetrac.DeepTrack.tracker import track_objects_with_splitting
        data = np.zeros((2, 1, 4, 4), dtype=int)
        result = track_objects_with_splitting(data)
        assert len(result) == 2


# ============================================================
# utils
# ============================================================

class TestComputeDaskQuantile:
    def test_shape_removes_time(self):
        from ocetrac.DeepTrack.utils import compute_dask_quantile
        da = _da(np.random.rand(10, 3, 4, 5)).chunk({"time": 5})
        q  = compute_dask_quantile(da, q=0.9)
        assert q.shape == (3, 4, 5)
        assert "time" not in q.dims

    def test_median_value(self):
        from ocetrac.DeepTrack.utils import compute_dask_quantile
        data = np.arange(10, dtype=float).reshape(10, 1, 1, 1)
        da   = _da(data).chunk({"time": 5})
        q    = compute_dask_quantile(da, q=0.5).compute().values.item()
        assert abs(q - 4.5) < 0.5


# ============================================================
# DeepTracker integration
# ============================================================

class TestDeepTrackerIntegration:
    def _tracker(self, T=5, Z=3, Y=12, X=12):
        da = _tracker_da(T=T, Z=Z, Y=Y, X=X)
        cv = np.ones((Z, Y, X)) * 1e6
        return da, cv

    def test_run_returns_dataarray(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        result = DeepTracker(da, radius=1, min_area_cells=1,
                             frac_filter=0.).run(cell_volume=cv)
        assert isinstance(result, xr.DataArray)
        assert result.shape == da.shape

    def test_result_background_is_nan(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        arr = t.result.values
        # Background cells (originally 0) must be NaN in result
        assert not (arr == 0).any()

    def test_n_events_positive(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        assert t.n_events() >= 1

    def test_event_duration_dict(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker(T=5)
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        durations = t.event_duration()
        assert isinstance(durations, dict)
        for eid, dur in durations.items():
            assert isinstance(eid, int) and isinstance(dur, int) and dur > 0

    def test_summary_runs_without_error(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        t.summary()

    def test_repr_before_run(self):
        from ocetrac.DeepTrack import DeepTracker
        da, _ = self._tracker()
        assert "(not run yet)" in repr(DeepTracker(da))

    def test_repr_after_run(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        t.run(cell_volume=cv)
        assert "DeepTracker(" in repr(t)
        assert "(not run yet)" not in repr(t)

    def test_postprocess_raises_without_track(self):
        from ocetrac.DeepTrack import DeepTracker
        da, _ = self._tracker()
        with pytest.raises(RuntimeError, match="track"):
            DeepTracker(da).postprocess()

    def test_n_events_raises_before_run(self):
        from ocetrac.DeepTrack import DeepTracker
        da, _ = self._tracker()
        with pytest.raises(RuntimeError):
            DeepTracker(da).n_events()

    def test_chaining_returns_self(self):
        from ocetrac.DeepTrack import DeepTracker
        da, cv = self._tracker()
        t = DeepTracker(da, radius=1, min_area_cells=1, frac_filter=0.)
        assert t.clean() is t
        assert t.label() is t
        assert t.connect_depth() is t
        assert t.prefilter() is t
        assert t.track(cell_volume=cv) is t

    def test_step_by_step_matches_run(self):
        """Running steps individually must produce the same result as .run()."""
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