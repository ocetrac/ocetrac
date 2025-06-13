import cftime
import numpy as np
import pytest
import xarray as xr

import ocetrac


@pytest.fixture
def example_data():
    x0 = [180, 225, 360, 80, 1, 360, 1]
    y0 = [0, 20, -50, 40, -50, 40, 40]
    sigma0 = [15, 25, 30, 10, 30, 15, 10]

    lon = np.arange(0, 360) + 0.5
    lat = np.arange(-90, 90) + 0.5
    x, y = np.meshgrid(lon, lat)
    timedim = "time"
    xdim = "lon"
    ydim = "lat"

    def make_blobs(x0, y0, sigma0):
        blob = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma0**2))
        return blob

    features = {}
    for i in range(len(x0)):
        features[i] = make_blobs(x0[i], y0[i], sigma0[i])

    first_image = features[0] + features[1] + features[3] - 0.5

    da = xr.DataArray(
        first_image[np.newaxis, :, :],
        dims=[timedim, ydim, xdim],
        coords={timedim: [1], ydim: lat, xdim: lon},
    )

    da_shift_01 = da.shift(lon=0, lat=-20, fill_value=-0.5)
    da_shift_02 = da.shift(lon=0, lat=-40, fill_value=-0.5) + (
        features[2] + features[4] + features[5] + features[6]
    )
    da_shift_03 = da.shift(lon=0, lat=-40, fill_value=-0.5) + (
        features[2] + features[5] + features[6]
    )

    Anom = xr.concat(
        (
            da,
            da_shift_01,
            da_shift_02,
            da_shift_03,
        ),
        dim="time",
    )

    Anom["time"] = np.arange(1, 5)
    Anom = Anom.where(Anom > 0, drop=False, other=0)

    mask = xr.DataArray(np.ones(Anom[0, :, :].shape), coords=Anom[0, :, :].coords)
    mask[60:90, 120:190] = 0

    radius = 8
    min_size_quartile = 0
    positive = True

    tracker = ocetrac.Tracker(
        Anom.chunk({"time": 1}),
        mask,
        radius,
        min_size_quartile,
        timedim,
        xdim,
        ydim,
        positive,
    )
    blobs = tracker.track()

    return blobs, Anom


def test_convert_lons(example_data):
    blobs, Anom = example_data
    new_Anom = ocetrac.measures.utils.lons_to_180(Anom)
    assert list(new_Anom["lon"].values) == list(np.linspace(-179.5, 179.5, 360))
    new_Anom2 = ocetrac.measures.utils.lons_to_360(new_Anom)
    assert list(new_Anom2["lon"].values) == list(np.linspace(0.5, 359.5, 360))


@pytest.mark.parametrize("obj_id", [1, 3, 5, np.nan])
def test_get_object_masks(example_data, obj_id):
    blobs, Anom = example_data
    if obj_id not in [1, 2, 3, 4]:
        with pytest.warns(
            UserWarning,
            match=rf'object_id "{obj_id}" not found in blobs. Returning zeros.',
        ):
            object_binary, object_intensity = ocetrac.measures.utils.get_object_masks(
                blobs, Anom, object_id=obj_id
            )
    else:
        object_binary, object_intensity = ocetrac.measures.utils.get_object_masks(
            blobs, Anom, object_id=obj_id
        )
        assert isinstance(object_intensity, xr.DataArray)
        assert len(object_binary.time) > 0
        assert list(blobs.where(blobs == obj_id, drop=True).time.values) == list(
            object_binary.time.values
        )


@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_shape_measures(example_data, obj_id):
    blobs, Anom = example_data
    object_binary, object_intensity = ocetrac.measures.utils.get_object_masks(
        blobs, Anom, object_id=obj_id
    )
    shape_measures = ocetrac.measures.utils.run_shape_measures(
        object_binary, lat_resolution_value=110.574, lon_resolution_value=111.320
    )
    result_keys = [
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
    result_lengths = [
        len(object_binary.time),
        len(object_binary.time),
        1,
        1,
        len(object_binary.time) - 1,
        len(object_binary.time) - 1,
        len(object_binary.time),
        len(object_binary.time),
        len(object_binary.time),
    ]
    assert list(shape_measures.keys()) == result_keys
    assert [
        len(item[1]) if type(item[1]) != np.float64 else 1
        for item in list(shape_measures.items())
    ] == result_lengths


@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_motion_measures(example_data, obj_id):
    blobs, Anom = example_data
    object_binary, object_intensity = ocetrac.measures.utils.get_object_masks(
        blobs, Anom, object_id=obj_id
    )
    results = ocetrac.measures.utils.run_motion_measures(
        object_binary, object_intensity
    )
    result_keys = [
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
    result_lengths = [
        len(object_binary.time),
        len(object_binary.time),
        len(object_binary.time),
        len(object_binary.time) - 1,
        len(object_binary.time),
        len(object_binary.time),
        len(object_binary.time),
        len(object_binary.time) - 1,
        5,
        5,
        2,
        2,
    ]
    assert list(results.keys()) == result_keys
    assert [
        len(item[1]) if type(item[1]) != np.float64 else 1
        for item in list(results.items())
    ] == result_lengths


@pytest.mark.parametrize("obj_id", [1, 3])
def test_run_temporal_measures(example_data, obj_id):
    # Minimal testing here. Should include tests of all time dimension options
    blobs, Anom = example_data
    object_binary, object_intensity = ocetrac.measures.utils.get_object_masks(
        blobs, Anom, object_id=obj_id
    )
    with pytest.raises(ValueError):
        results = ocetrac.measures.utils.run_temporal_measures(
            object_binary.sel(time=slice(0, 0))
        )
    results = ocetrac.measures.utils.run_temporal_measures(object_binary)
    result_keys = ["initial_detection_date", "duration"]
    assert list(results.keys()) == result_keys
    assert results["initial_detection_date"] == object_binary.time[0]


@pytest.mark.parametrize("quantile_threshold", [0.5, 0.9])
@pytest.mark.parametrize("obj_id", [1, 3])
def test_calculate_intensity_metrics(example_data, obj_id, quantile_threshold):
    blobs, Anom = example_data
    object_binary, object_intensity = ocetrac.measures.utils.get_object_masks(
        blobs, Anom, object_id=obj_id
    )
    results = ocetrac.measures.calculate_intensity_metrics(
        object_intensity, quantile_threshold
    )
    result_keys = [
        "cumulative_intensity",
        "mean_intensity_timeseries",
        "mean_intensity",
        "max_intensity_timeseries",
        "max_intensity",
        "std_intensity_timeseries",
        "std_intensity",
        f"percentile_{int(quantile_threshold*100)}_intensity_timeseries",
        f"percentile_{int(quantile_threshold*100)}_intensity",
        "quantile_threshold_used",
    ]
    result_lengths = [
        len(object_binary.time),
        len(object_binary.time),
        1,
        len(object_binary.time),
        1,
        len(object_binary.time),
        1,
        len(object_binary.time),
        1,
        1,
    ]
    assert list(results.keys()) == result_keys
    assert [
        len(item[1]) if type(item[1]) != float else 1 for item in list(results.items())
    ] == result_lengths


@pytest.mark.parametrize("obj_ids", [[0], [1], [1, 2]])
def test_process_objects_and_calculate_measures(example_data, obj_ids):
    blobs, Anom = example_data
    valid_object_ids = np.unique(blobs)
    if all([i in valid_object_ids[~np.isnan(valid_object_ids)] for i in obj_ids]):
        results = ocetrac.measures.utils.process_objects_and_calculate_measures(
            obj_ids, blobs, Anom
        )
        result_keys = [
            "shape_measures",
            "motion_measures",
            "temporal_measures",
            "intensity_measures",
        ]
        assert list(results.keys()) == obj_ids
        for obj_id in obj_ids:
            assert list(results[obj_id].keys()) == result_keys
    else:
        with pytest.raises(ValueError):
            results = ocetrac.measures.utils.process_objects_and_calculate_measures(
                obj_ids, blobs, Anom
            )


# def test_calculate_anomalies_trend_features(example_data):
#     pass
