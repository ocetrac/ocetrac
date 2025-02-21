import xarray as xr
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage import img_as_float
from skimage.morphology import convex_hull_image
from skimage.measure import label as label_np, regionprops, find_contours
from haversine import haversine, Unit
from typing import Dict, Optional, List, Tuple
import numpy as np

def _get_labels(binary_images: np.ndarray) -> np.ndarray:
    """
    Labels connected regions in binary images.

    Parameters
    ----------
    binary_images : np.ndarray
        A 2D or 3D NumPy array representing binary images. Pixels with a value of 1 are considered
        part of a region, while pixels with a value of 0 are considered background.

    Returns
    -------
    np.ndarray
        A labeled array where each connected region is assigned a unique integer label.
        Background pixels are labeled as 0.

    Notes
    -----
    This function replaces NaN values with 0 and ensures the input array contains only finite values
    before applying the labeling algorithm.
    """
    # Replace NaN values with 0 (background)
    binary_images = np.nan_to_num(binary_images, nan=0)
    
    # Ensure the array contains only valid numeric values
    binary_images = np.where(np.isfinite(binary_images), binary_images, 0)
    
    return label_np(binary_images, background=0)

def _get_centroids(sub_labels: xr.DataArray) -> List[Tuple[float, float]]:
    """
    Calculates the centroids of labeled regions.

    Parameters
    ----------
    sub_labels : xr.DataArray
        A labeled xarray DataArray where each connected region is assigned a unique integer label.
        The DataArray must have 'lat' and 'lon' coordinates representing latitude and longitude.

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples, where each tuple contains the centroid coordinates (latitude, longitude)
        of a labeled region.

    Notes
    -----
    This function uses `skimage.measure.regionprops` to compute the centroids of labeled regions.
    NaN values in the input array are replaced with 0 before processing.
    """
    # Replace NaN values with 0 (background)
    labeled_array = sub_labels.fillna(0).astype(int).values

    # Compute centroids using regionprops
    props = regionprops(labeled_array)
    centroids = [
        (float(sub_labels.lat[round(p.centroid[0])].values),
         float(sub_labels.lon[round(p.centroid[1])].values))
        for p in props
    ]
    return centroids

def centroids_per_timestep(one_obj: xr.DataArray, timestep: int) -> List[Tuple[float, float]]:
    """
    Calculates centroids for each timestep, handling edge cases only if longitude spans 0 to 360.

    Parameters
    ----------
    one_obj : xr.DataArray
        An xarray DataArray containing labeled regions for each timestep. The DataArray must have
        dimensions ('time', 'lat', 'lon') and coordinates for latitude and longitude.
    timestep : int
        The timestep to process. This corresponds to the index along the 'time' dimension.

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples, where each tuple contains the centroid coordinates (latitude, longitude)
        of a labeled region for the specified timestep.

    Notes
    -----
    This function handles edge cases where regions wrap around the longitudinal boundary (e.g., from
    0 to 360 degrees). If the longitude range spans 0 to 360, regions touching the left and right
    edges are merged before computing centroids.
    """
    # Step 1: Extract the timestep of interest
    timestep_of_interest = one_obj.labels.isel(time=timestep)
    
    # Step 2: Label the regions in the timestep
    sub_labels = _get_labels(timestep_of_interest.values)
    sub_labels = xr.DataArray(sub_labels, dims=timestep_of_interest.dims, coords=timestep_of_interest.coords)
    sub_labels = sub_labels.where(timestep_of_interest > 0, drop=False, other=np.nan)

    # Step 3: Check if longitude spans 0 to 360
    handle_edge_cases = (one_obj.lon[0].item() == 0 and 
                         one_obj.lon[-1].item() == 360)

    # Step 4: Identify edge labels (regions that touch the left or right edge)
    if handle_edge_cases:
        edge_right_sub_labels = np.unique(sub_labels[:, -1:].values[~np.isnan(sub_labels[:, -1:].values)])
        edge_left_sub_labels = np.unique(sub_labels[:, :1].values[~np.isnan(sub_labels[:, :1].values)])
        edge_labels = np.unique(np.concatenate((edge_right_sub_labels, edge_left_sub_labels)))
    else:
        edge_labels = np.array([])  # No edge labels to handle

    # Step 5: Identify non-edge labels
    nonedgecases = np.setdiff1d(np.unique(sub_labels), edge_labels)
    nonedgecases = nonedgecases[~np.isnan(nonedgecases)]

    # Step 6: Compute centroids for non-edge regions
    centroid_list = []
    for i in nonedgecases:
        sub_labels_nonedgecases = sub_labels.where(sub_labels == i, drop=False, other=np.nan)
        centroid_list.append(_get_centroids(sub_labels_nonedgecases))

    # Step 7: Handle edge regions (merge regions that wrap around the longitudinal boundary)
    if handle_edge_cases:
        for i in edge_left_sub_labels:
            sub_labels_left = sub_labels.where(sub_labels == i, drop=True)
            lon_edge = sub_labels_left[:, -1:].lon.item()
            sub_labels_left.coords['lon'] = (sub_labels_left.coords['lon'] + 360)  # Shift left edge to the right

            for j in edge_right_sub_labels:
                sub_labels_right = sub_labels.where(sub_labels == j, drop=False, other=np.nan)
                east = sub_labels_right.where(sub_labels_right.lon > lon_edge, drop=True)
                append_east = xr.concat([east.where(east.lon >= lon_edge, drop=True), sub_labels_left], dim="lon")
                append_east_binarized = xr.where(append_east > 0, 1, np.nan)
                sub_labels_new = _get_labels(append_east_binarized.values)
                sub_labels_new = xr.DataArray(sub_labels_new, dims=append_east_binarized.dims, coords=append_east_binarized.coords)

                # Check if the merged region is valid (contains only one or two labels)
                if len(np.unique(sub_labels_new)) <= 2:
                    sub_labels_new = sub_labels_new.where(append_east_binarized > 0, drop=False, other=np.nan)
                    centroid_list.append(_get_centroids(sub_labels_new))

    # Step 8: Flatten the list of centroids and remove duplicates
    flat_centroid_list = list(set([item for sublist in centroid_list for item in sublist]))
    return flat_centroid_list

def displacement(one_obj_labels: xr.DataArray) -> Tuple[List[Tuple[float, float]], List[float], List[Tuple[float, float]]]:
    """
    Calculate the displacement of centroids and center of mass over time.

    Parameters
    ----------
    one_obj_labels : xr.DataArray
        An xarray DataArray containing labeled regions for each timestep. The DataArray must have
        dimensions ('time', 'lat', 'lon') and coordinates for latitude and longitude.

    Returns
    -------
    Tuple[List[Tuple[float, float]], List[float], List[Tuple[float, float]]]
        A tuple containing:
        - centroid_coords: A list of tuples with the centroid coordinates (latitude, longitude) for each timestep.
        - com_displacements: A list of distances (in kilometers) between centroid coordinates for consecutive timesteps.
        - centroid_xrcoords: A list of xarray coordinates for the centroid for each timestep.

    Notes
    -----
    This function computes the centroids and their displacements over time. It uses the Haversine formula
    to calculate distances between consecutive centroids, accounting for the Earth's curvature.
    """
    # Initialize lists to store results
    centroid_list = []
    centroid_xrcoords_ls = []
    
    # Iterate over each timestep
    for i in range(one_obj_labels.shape[0]):
        # Binarize the labels (1 for object, NaN for background)
        labels_binarized = xr.where(one_obj_labels[i, :, :] > 0, 1, np.nan)
        
        # Calculate centroid using center of mass
        img_cent = labels_binarized.fillna(0)
        centroid_list.append(ndimage.center_of_mass(img_cent.data))
        
        # Calculate centroid coordinates in xarray format
        img_cent_xr_coords = _get_center_of_mass(labels_binarized)
        centroid_xrcoords_ls.append(img_cent_xr_coords[0])
    
    # Extract latitude and longitude values from centroids
    y_val_cent, x_val_cent = zip(*centroid_list)
    
    # Create interpolation functions for latitude and longitude
    lat_range = [one_obj_labels.lat[0].item(), one_obj_labels.lat[-1].item()]
    lon_range = [one_obj_labels.lon[0].item(), one_obj_labels.lon[-1].item()]
    convert_lat_range = interp1d([0, one_obj_labels.lat.shape[0]], lat_range)
    convert_long_range = interp1d([0, one_obj_labels.lon.shape[0]], lon_range)
    
    # Convert centroid coordinates to latitude and longitude
    coords_cent = list(zip(convert_lat_range(y_val_cent), convert_long_range(x_val_cent)))

    convert_long_range_ew = interp1d([0,one_obj.lon.shape[0]],
                                     [one_obj.lon[0].item()-180,360-one_obj.lon[-1].item()])
    coords_cent_w_ew = list(zip(convert_lat_range(y_val_cent), 
                                convert_long_range_ew(x_val_cent)))
    
    # Calculate displacements between consecutive centroids
    distance_cent_ls = []
    for i in range(len(coords_cent) - 1):
        distance_cent = haversine(coords_cent_w_ew[i], coords_cent_w_ew[i + 1], Unit.KILOMETERS)
        distance_cent_ls.append(distance_cent)
    
    return centroid_list, distance_cent_ls, centroid_xrcoords_ls

def calculate_velocity(
    centroid_coords: List[Tuple[float, float]], 
    time_interval: float
) -> List[Tuple[float, float]]:
    """
    Calculate the velocity of centroids over time.

    Parameters
    ----------
    centroid_coords : List[Tuple[float, float]]
        List of (latitude, longitude) coordinates for centroids over time.
    time_interval : float
        Time interval between consecutive centroids (in hours, days, etc.).

    Returns
    -------
    List[Tuple[float, float]]
        List of velocity vectors (v_lat, v_lon) for each time step.
    """
    velocities = []
    for i in range(1, len(centroid_coords)):
        # Calculate displacement in latitude and longitude
        d_lat = centroid_coords[i][0] - centroid_coords[i-1][0]
        d_lon = centroid_coords[i][1] - centroid_coords[i-1][1]
        
        # Calculate velocity components
        v_lat = d_lat / time_interval
        v_lon = d_lon / time_interval
        
        velocities.append((v_lat, v_lon))
    
    return velocities

def calculate_acceleration(
    velocities: List[Tuple[float, float]], 
    time_interval: float
) -> List[Tuple[float, float]]:
    """
    Calculate the acceleration of centroids over time.

    Parameters
    ----------
    velocities : List[Tuple[float, float]]
        List of velocity vectors (v_lat, v_lon) over time.
    time_interval : float
        Time interval between consecutive velocities (in hours, days, etc.).

    Returns
    -------
    List[Tuple[float, float]]
        List of acceleration vectors (a_lat, a_lon) for each time step.
    """
    accelerations = []
    for i in range(1, len(velocities)):
        # Calculate change in velocity components
        dv_lat = velocities[i][0] - velocities[i-1][0]
        dv_lon = velocities[i][1] - velocities[i-1][1]
        
        # Calculate acceleration components
        a_lat = dv_lat / time_interval
        a_lon = dv_lon / time_interval
        
        accelerations.append((a_lat, a_lon))
    
    return accelerations