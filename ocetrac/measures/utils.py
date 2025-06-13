"""
utils.py

Utility functions for analyzing motion, shape, intensity, and temporal characteristics of geospatial objects.
"""

import warnings

import numpy as np
import xarray as xr

# from ocetrac.measures import (
from . import (
    MotionMeasures,
    ShapeMeasures,
    calculate_intensity_metrics,
    get_duration,
    get_initial_detection_time,
    plot_displacement,
)


def lons_to_360(data, coord="lon"):
    """
    Converts longitude coordinates from (-180, 180) to (0, 360).

    Parameters:
    data : xarray.DataArray or xarray.Dataset
        Array with longitude coordinate.
    coord : str
        Name of longitude coordinate. Defaults to "lon".

    Returns:
    xarray.DataArray or xarray.Dataset
        DataArray/Dataset with original longitude coordinate now between (0, 360).
    """
    if coord not in data.dims:
        raise ValueError("Longitude coordinate must be in data.dims.")
    data.coords[coord] = (360 + (data.coords[coord] % 360)) % 360
    data = data.sortby(data[coord])
    return data


def lons_to_180(data, coord="lon"):
    """
    Converts longitude coordinates from (0, 360) to (-180, 180).

    Parameters:
    data : xarray.DataArray or xarray.Dataset
        DataArray/Dataset with longitude coordinate.
    coord : str
        Name of longitude coordinate. Defaults to "lon".

    Returns:
    xarray.DataArray or xarray.Dataset
        DataArray/Dataset with original longitude coordinate now between (-180, 180).
    """
    if coord not in data.dims:
        raise ValueError("Longitude coordinate must be in data.dims.")
    data.coords[coord] = (data.coords[coord] + 180) % 360 - 180
    data = data.sortby(data[coord])
    return data


def get_object_masks(blobs, var_notrend, object_id):
    """
    Extract labels and masked anomalies for a specific object ID.

    Parameters:
    -----------
    blobs : xarray.DataArray
        Array containing object IDs
    var_notrend : xarray.DataArray
        Array of anomalies (detrended variable)
    object_id : int or float
        The object ID to analyze

    Returns:
    --------
    tuple:
        (labels, masked_anomalies_nan) where:
        - labels is a binary mask of the object (1 where object exists, 0 otherwise)
        - masked_anomalies_nan contains anomaly values only where object exists (nan elsewhere)
    """
    # Check object exists
    valid_object_ids = np.unique(blobs)
    if object_id in valid_object_ids[~np.isnan(valid_object_ids)]:
        pass
    else:
        warnings.warn(f'object_id "{object_id}" not found in blobs. Returning zeros.')

    # Calculate time steps where object exists
    object_count_per_time = (blobs == object_id).sum(dim=["lat", "lon"])
    true_time_steps = object_count_per_time.time.where(
        object_count_per_time > 0, drop=True
    )
    one_obj = blobs.sel(time=true_time_steps.time)

    one_obj_anom = var_notrend.sel(time=true_time_steps.time)

    only_one_obj = xr.where(one_obj == object_id, 1.0, 0)

    masked_one_obj_anomalies = var_notrend * only_one_obj

    # INPUT TO MEASURES SUBMODULE
    masked_one_obj_anomalies_nan = xr.where(  # Will use this as the intensity
        masked_one_obj_anomalies > 0.0, masked_one_obj_anomalies, np.nan
    )

    one_obj["labels"] = only_one_obj  # This is the binary event mask
    one_obj = xr.where(one_obj.labels == 0.0, np.nan, 1.0)
    one_obj["labels"] = one_obj

    return one_obj, masked_one_obj_anomalies_nan


def run_shape_measures(one_obj, lon_resolution_value, lat_resolution_value):
    """
    Calculates various shape measures for a given object using the Ocetrac library.

    Parameters
    ----------
    one_obj : xarray.DataArray
        An xarray DataArray representing the binary mask of the object
        (1 for object pixels, NaN otherwise). This DataArray should have
        'time', 'lat', and 'lon' dimensions.
    lon_resolution_value : float
        The resolution of longitude in kilometers per degree.
    lat_resolution_value : float
        The resolution of latitude in kilometers per degree.

    Returns
    -------
    dict
        A dictionary containing the calculated shape measures:
        'perimeters' : Perimeters of the object at each time step.
        'spatial_extents' : Spatial extents (areas) of the object at each time step.
        'max_spatial_extent' : Maximum spatial extent over all time steps.
        'mean_spatial_extent' : Mean spatial extent over all time steps.
        'complement_to_deformation' : Complement to deformation at each time step.
        'deformation' : Deformation at each time step.
        'convex_hull_areas' : Areas of the convex hull of the object at each time step.
        'ratio_convexhullarea_vs_area' : Ratio of convex hull area to object area at each time step.
        'circularity' : Circularity of the object at each time step.
    """
    # Instantiate the class
    metrics = ShapeMeasures(
        lat_resolution=lat_resolution_value,
        lon_resolution=lon_resolution_value,
        use_decorators=False,
    )

    # Area
    spatial_extent_data = metrics.calc_spatial_extents(one_obj)

    # Perimeter
    perimeters = metrics.calc_perimeter(one_obj)

    # Complement to deformation
    coords_full = spatial_extent_data["coords_full"]
    spatial_extents = spatial_extent_data["spatial_extents"]
    comp_to_deform = metrics.calc_complement_to_deformation(
        coords_full, spatial_extents
    )

    # Deformation
    deform = metrics.calc_deformation(comp_to_deform)

    # Convex hull area vs. object area
    (
        convex_hull_areas,
        ratio_convexhullarea_vs_area,
    ) = metrics.calc_ratio_convexhullarea_vs_area(one_obj)

    # Circularity
    circularity = metrics.calc_circularity(spatial_extents, perimeters)

    return {
        "perimeters": perimeters,
        "spatial_extents": spatial_extents,
        "max_spatial_extent": spatial_extent_data["max_spatial_extent"],
        "mean_spatial_extent": spatial_extent_data["mean_spatial_extent"],
        "complement_to_deformation": comp_to_deform,
        "deformation": deform,
        "convex_hull_areas": convex_hull_areas,
        "ratio_convexhullarea_vs_area": ratio_convexhullarea_vs_area,
        "circularity": circularity,
    }


def run_motion_measures(one_obj, masked_one_obj_anomalies_data):
    """
    Calculates various motion measures for a given object and its associated
    masked anomaly data using the Ocetrac library.

    Parameters
    ----------
    one_obj : xarray.DataArray
        An xarray DataArray representing the binary mask of the object
        (1 for object pixels, NaN otherwise). This DataArray should have
        'time', 'lat', and 'lon' dimensions.
    masked_one_obj_anomalies_data : xarray.DataArray
        An xarray DataArray representing the anomaly data for the object,
        masked to the object's extent. This DataArray should have
        'time', 'lat', and 'lon' dimensions.

    Returns
    -------
    dict
        A dictionary containing the calculated motion measures:
        'num_centroids_per_timestep' : List of the number of centroids found at each time step.
        'centroid_coords' : List of lists, where each inner list contains the centroid coordinates
                            ([lon, lat]) for each object instance at a given time step.
        'centroid_path' : The path traced by the centroid of the object over time (list of coordinates).
        'centroid_displacements_km' : List of the displacement in kilometers between consecutive centroid positions.
        'num_coms_per_timestep' : List of the number of centers of mass (COMs) found at each time step.
        'com_coords' : List of lists, where each inner list contains the COM coordinates
                       ([lon, lat]) for each object instance at a given time step.
        'com_path' : The path traced by the COM of the object over time (list of coordinates).
        'com_displacements_km' : List of the displacement in kilometers between consecutive COM positions.
        'com_directionality' : Dictionary containing COM directionality statistics (e.g., mean direction).
        'centroid_directionality' : Dictionary containing centroid directionality statistics (e.g., mean direction).
        'centroid_displacement_plot_data' : Tuple containing data suitable for plotting centroid displacement (path, mask).
        'com_displacement_plot_data' : Tuple containing data suitable for plotting COM displacement (path, masked anomalies).
    """
    motion = MotionMeasures(use_decorators=False)
    # Calculate centroids
    num_centroids = []
    centroid_coords = []
    for timestep in range(one_obj.time.shape[0]):
        centroids = motion.calculate_centroids(one_obj.labels, timestep)
        num_centroids.append(len(centroids))
        centroid_coords.append(centroids)

    # Calculate centroid displacements
    centroid_path, centroid_displacements = motion.calculate_centroid_displacement(
        one_obj.labels
    )

    # Calculate centers of mass
    num_coms = []
    com_coords = []
    for timestep in range(one_obj.time.shape[0]):
        coms = motion.calculate_coms(
            one_obj.labels, masked_one_obj_anomalies_data, timestep
        )
        num_coms.append(len(coms))
        com_coords.append(coms)

    # Calculate COM displacements
    com_path, com_displacements = motion.calculate_com_displacement(
        masked_one_obj_anomalies_data
    )

    # Calculate directionality
    com_dir = motion.calculate_directionality(com_path)
    centroid_dir = motion.calculate_directionality(centroid_path)

    return {
        "num_centroids_per_timestep": num_centroids,
        "centroid_coords": centroid_coords,
        "centroid_path": centroid_path,
        "centroid_displacements_km": centroid_displacements,
        "num_coms_per_timestep": num_coms,
        "com_coords": com_coords,
        "com_path": com_path,
        "com_displacements_km": com_displacements,
        "com_directionality": com_dir,
        "centroid_directionality": centroid_dir,
        "centroid_displacement_plot_data": (centroid_path, one_obj.labels),
        "com_displacement_plot_data": (com_path, masked_one_obj_anomalies_data),
    }


def run_temporal_measures(one_obj):
    """
    Calculates temporal measures for a given object using the Ocetrac library.

    Parameters
    ----------
    one_obj : xarray.DataArray
        An xarray DataArray representing the binary mask of the object
        (1 for object pixels, NaN otherwise). This DataArray should have
        'time', 'lat', and 'lon' dimensions.

    Returns
    -------
    dict
        A dictionary containing the calculated temporal measures:
        'initial_detection_date' : The time of the first detection of the object.
        'duration' : The duration of the object's existence in time steps.
    """
    initial_detection_date = get_initial_detection_time(one_obj)
    duration = get_duration(one_obj)

    return {"initial_detection_date": initial_detection_date, "duration": duration}


def run_intensity_measures(masked_one_obj_anomalies_nan):
    """
    Calculates intensity measures for a given object's masked anomaly data
    using the Ocetrac library.

    Parameters
    ----------
    masked_one_obj_anomalies_nan : xarray.DataArray
        An xarray DataArray representing the anomaly data for the object,
        masked to the object's extent and with non-object values set to NaN.
        This DataArray should have 'time', 'lat', and 'lon' dimensions.

    Returns
    -------
    dict
        A dictionary containing the calculated intensity measures:
        'max_intensity' : The maximum intensity value observed for the object.
        'percentile_90_intensity_timeseries' : An array or list containing
                                               the 90th percentile intensity
                                               value at each time step.
    """
    masked_intensity = masked_one_obj_anomalies_nan
    intensity_metrics_default = calculate_intensity_metrics(masked_intensity)
    intensity_metrics_90th = calculate_intensity_metrics(
        masked_intensity, quantile_threshold=0.90
    )

    return {
        "max_intensity": intensity_metrics_default["max_intensity"],
        "percentile_90_intensity_timeseries": intensity_metrics_90th[
            "percentile_90_intensity_timeseries"
        ],
    }


def process_objects_and_calculate_measures(
    object_ids_to_process,
    blobs,
    var_notrend,
    run_shape=True,
    run_motion=True,
    run_temporal=True,
    run_intensity=True,
    lon_resolution_value=111.320,  # km per degree longitude
    lat_resolution_value=110.574,  # km per degree latitude
):
    """
    Processes multiple objects, calculates selected measures for each object,
    and returns results keyed by object ID.

    Parameters
    ----------
    object_ids_to_process : list of int or float
        A list of object IDs for which to calculate measures.
    blobs : xarray.DataArray
        Array containing object IDs
    var_notrend : xarray.DataArray
        Array of anomalies (detrended variable)
    run_shape : bool, optional
        Whether to calculate shape measures. Defaults to True.
    run_motion : bool, optional
        Whether to calculate motion measures. Defaults to True.
    run_temporal : bool, optional
        Whether to calculate temporal measures. Defaults to True.
    run_intensity : bool, optional
        Whether to calculate intensity measures. Defaults to True.
    lon_resolution_value : float, optional
        The resolution of longitude in kilometers per degree.
        Defaults to 111.320.
    lat_resolution_value : float, optional
        The resolution of latitude in kilometers per degree.
        Defaults to 110.574.

    Returns
    -------
    dict
        A nested dictionary where:
        - Keys are object IDs
        - Values are dictionaries containing measure results with keys:
            * 'shape_measures' (if run_shape=True)
            * 'motion_measures' (if run_motion=True)
            * 'temporal_measures' (if run_temporal=True)
            * 'intensity_measures' (if run_intensity=True)
        Returns an empty dictionary if no objects are processed successfully.
    """
    all_objects_measure_results = {}

    # Loop through each object ID
    for object_id in object_ids_to_process:
        try:
            # Check object exists
            valid_object_ids = np.unique(blobs)
            if object_id in valid_object_ids[~np.isnan(valid_object_ids)]:
                pass
            else:
                raise ValueError(
                    f'object_id "{object_id}" not found in blobs. Returning zeros.'
                )
            event_binary, event_intensity = get_object_masks(
                blobs, var_notrend, object_id=object_id
            )
            measure_results_for_current_object = {}

            if run_shape:
                measure_results_for_current_object[
                    "shape_measures"
                ] = run_shape_measures(
                    event_binary, lon_resolution_value, lat_resolution_value
                )
                print("Shape Measures complete.")

            if run_motion:
                measure_results_for_current_object[
                    "motion_measures"
                ] = run_motion_measures(event_binary, event_intensity)
                print("Motion Measures complete.")

            if run_temporal:
                measure_results_for_current_object[
                    "temporal_measures"
                ] = run_temporal_measures(event_binary)
                print("Temporal Measures complete.")

            if run_intensity:
                measure_results_for_current_object[
                    "intensity_measures"
                ] = run_intensity_measures(event_intensity)
                print("Intensity Measures complete.")

            all_objects_measure_results[object_id] = measure_results_for_current_object
        except Exception as e:
            print(f"An error occurred while processing ID {object_id}: {e}")
            raise e

    return all_objects_measure_results
