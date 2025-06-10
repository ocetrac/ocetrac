"""
motion_measures.py

Class for analyzing motion characteristics of geospatial objects:
- Centroid tracking
- Center of mass calculations
- Displacement, velocity and directionality metrics
"""

import xarray as xr
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage.measure import label as label_np, regionprops
from haversine import haversine, Unit
from typing import List, Tuple, Dict
from math import atan2, degrees

class MotionMeasures:
    """Class for calculating motion characteristics of labeled geospatial objects:
    
    Provides methods for tracking object movement through centroid and center of mass calculations,  
    displacement measures.
    """
    
    def __init__(self, use_decorators: bool = True):
        """
        Initialize motion metrics calculator.

        Parameters
        ----------
        use_decorators : bool, optional
            Enable/disable timing decorators (default: True)
        """
        self.use_decorators = use_decorators

    def _calculate_com(self, intensity_image: xr.DataArray) -> List[Tuple[float, float]]:
        """Calculate center of mass coordinates from intensity image.

        Parameters
        ----------
        intensity_image : xr.DataArray
            Intensity values with dimensions (lat, lon)
        
        Returns
        -------
        List[Tuple[float, float]]
            List containing single (latitude, longitude) center of mass coordinate
        
        Notes
        -----
        - Handles NaN values by filling with 0
        """
        img = intensity_image.fillna(0)
        com = ndimage.center_of_mass(img.data)
        centroid = (float(img.lat[round(com[0])].values),
                   float(img.lon[round(com[1])].values))
        
        if centroid[1] >= 359.75:
            centroid = (centroid[0], centroid[1] - 359.75)
        return [centroid]

    def _label_regions(self, binary_data: np.ndarray) -> np.ndarray:
        """Label connected regions in binary data.

        Parameters
        ----------
        binary_data : np.ndarray
            Binary data array with dimensions (lat, lon)
        
        Returns
        -------
        np.ndarray 
            Labeled regions with unique integer labels for each region
        """
        binary_data = np.nan_to_num(binary_data, nan=0)
        binary_data = np.where(np.isfinite(binary_data), binary_data, 0)
        return label_np(binary_data, background=0)

    def calculate_centroids(self, labels: xr.DataArray, timestep: int) -> List[Tuple[float, float]]:
        """
        Calculate centroids for all regions at specified timestep.

        Parameters
        ----------
        labels : xr.DataArray
            Labeled regions with dimensions (time, lat, lon)
        timestep : int
            Index of timestep to analyze
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (latitude, longitude) centroid coordinates for each region
        
        Notes
        -----
        - Handles NaN values by filling with 0
        - Handles edge cases for longitude wrapping
        """
        timestep_data = labels.isel(time=timestep)
        labeled = self._label_regions(timestep_data.values)
        labeled = xr.DataArray(labeled, dims=timestep_data.dims, coords=timestep_data.coords)
        labeled = labeled.where(timestep_data > 0, other=np.nan)

        # Handle edge cases for longitude wrapping
        if (labels.lon[0].item() == 0 and labels.lon[-1].item() == 360):
            edge_right = np.unique(labeled[:, -1:].values[~np.isnan(labeled[:, -1:].values)])
            edge_left = np.unique(labeled[:, :1].values[~np.isnan(labeled[:, :1].values)])
            edge_labels = np.unique(np.concatenate((edge_right, edge_left)))
        else:
            edge_labels = np.array([])

        non_edge = np.setdiff1d(np.unique(labeled), edge_labels)
        non_edge = non_edge[~np.isnan(non_edge)]

        centroids = []
        for i in non_edge:
            region = labeled.where(labeled == i, other=np.nan)
            props = regionprops(region.fillna(0).astype(int).values)
            for p in props:
                centroids.append(
                    (float(labeled.lat[round(p.centroid[0])].values),
                    float(labeled.lon[round(p.centroid[1])].values))
                )

        # Handle edge regions (longitude wrapping)
        if len(edge_labels) > 0:
            for i in edge_left:
                left_region = labeled.where(labeled == i, drop=True)
                lon_edge = left_region[:, -1:].lon.item()
                left_region.coords['lon'] = (left_region.coords['lon'] + 360)

                for j in edge_right:
                    right_region = labeled.where(labeled == j, other=np.nan)
                    east = right_region.where(right_region.lon > lon_edge, drop=True)
                    merged = xr.concat([east.where(east.lon >= lon_edge, drop=True), 
                                      left_region], dim="lon")
                    merged_bin = xr.where(merged > 0, 1, np.nan)
                    labels_new = self._label_regions(merged_bin.values)
                    labels_new = xr.DataArray(labels_new, dims=merged_bin.dims, 
                                            coords=merged_bin.coords)
                    if len(np.unique(labels_new)) <= 2:
                        labels_new = labels_new.where(merged_bin > 0, other=np.nan)
                        props = regionprops(labels_new.fillna(0).astype(int).values)
                        for p in props:
                            row, col = p.centroid
                            lat_cent = np.interp(row, np.arange(labeled.sizes['lat']), labeled.lat.values)
                            lon_cent = np.interp(col, np.arange(labeled.sizes['lon']), labeled.lon.values)
                            centroids.append((float(lat_cent), float(lon_cent)))

                            # centroids.append(
                            #     (float(labels_new.lat[round(p.centroid[0])].values),
                            #     float(labels_new.lon[round(p.centroid[1])].values))
                            # )

        return list(set(centroids))

    def calculate_coms(self, labels: xr.DataArray, intensity: xr.DataArray, 
                      timestep: int) -> List[Tuple[float, float]]:
        """
        Calculate centers of mass for intensity-weighted regions.

        Parameters
        ----------
        labels : xr.DataArray
            Labeled regions with dimensions (time, lat, lon)
        intensity : xr.DataArray
            Intensity values with dimensions (time, lat, lon)
        timestep : int
            Index of timestep to analyze

        Returns
        -------
        List[Tuple[float, float]]
            List of (latitude, longitude) center of mass coordinates
        """
        labels_ts = labels[timestep, :, :]
        intensity_ts = intensity[timestep, :, :]
        
        labeled = self._label_regions(labels_ts.values)
        labeled = xr.DataArray(labeled, dims=labels_ts.dims, coords=labels_ts.coords)
        labeled = labeled.where(labels_ts > 0, other=np.nan)

        # Edge detection
        if (labels.lon[0].item() == 0 and labels.lon[-1].item() == 360):
            edge_right = np.unique(labeled[:, -1:].values[~np.isnan(labeled[:, -1:].values)])
            edge_left = np.unique(labeled[:, :1].values[~np.isnan(labeled[:, :1].values)])
            edge_labels = np.unique(np.concatenate((edge_right, edge_left)))
        else:
            edge_labels = np.array([])

        non_edge = np.setdiff1d(np.unique(labeled), edge_labels)
        non_edge = non_edge[~np.isnan(non_edge)]

        coms = []
        for i in non_edge:
            region_intensity = xr.where(labeled == i, intensity_ts, np.nan)
            coms.append(self._calculate_com(region_intensity)[0])
            
        # Handle edge regions with intensity
        if len(edge_labels) > 0:
            for i in edge_left:
                left_region = labeled.where(labeled == i, drop=True)
                lon_edge = left_region[:, -1:].lon.item()
                if lon_edge < 358.75:
                    left_intensity = intensity_ts.where((intensity_ts.lon <= lon_edge), drop=True)
                    left_region.coords['lon'] = (left_region.coords['lon'] + 360)
                    left_intensity.coords['lon'] = (left_intensity.coords['lon'] + 360)
                    
                    for j in edge_right:
                        right_intensity = intensity_ts.where(labeled == j, other=np.nan)
                        east_intensity = right_intensity.where(right_intensity.lon > lon_edge, drop=True)
                        merged_intensity = xr.concat([east_intensity.where(east_intensity.lon >= lon_edge, drop=True), 
                                                    left_intensity], dim="lon")
                        merged_bin = xr.where(merged_intensity.notnull(), 1, np.nan)
                        labels_new = self._label_regions(merged_bin.values)
                        labels_new = xr.DataArray(labels_new, dims=merged_bin.dims, 
                                                coords=merged_bin.coords)
                        if len(np.unique(labels_new)) <= 2:
                            merged_intensity = merged_intensity.where(merged_bin > 0, other=np.nan)
                            coms.append(self._calculate_com(merged_intensity)[0])
        return coms

    def calculate_centroid_displacement(self, labels: xr.DataArray) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate centroid displacements over time.

        Parameters
        ----------
        labels : xr.DataArray
            Labeled regions with dimensions (time, lat, lon)

        Returns
        -------
        Tuple[List[Tuple[float, float]], List[float]]
            - List of (lat, lon) centroid coordinates per timestep
            - List of displacement distances between timesteps (km)
        """
        centroids = []
        for i in range(labels.shape[0]):
            centroids.append(self._calculate_com(labels[i, :, :])[0])
        
        # Calculate displacements
        displacements = []
        for i in range(len(centroids) - 1):
            dist = haversine(centroids[i], centroids[i + 1], Unit.KILOMETERS)
            displacements.append(dist)
            
        return centroids, displacements
        
    def calculate_com_displacement(self, intensity: xr.DataArray) -> Tuple[List[Tuple[float, float]], List[float]]:
        """
        Calculate center of mass displacements over time.

        Parameters
        ----------
        intensity : xr.DataArray  
            Intensity values with dimensions (time, lat, lon)

        Returns
        -------
        Tuple[List[Tuple[float, float]], List[float]]
            - List of (lat, lon) COM coordinates per timestep  
            - List of displacement distances between timesteps (km)
        """
        coms = []
        for i in range(intensity.shape[0]):
            coms.append(self._calculate_com(intensity[i, :, :])[0])
        
        # Calculate displacements
        displacements = []
        for i in range(len(coms) - 1):
            dist = haversine(coms[i], coms[i + 1], Unit.KILOMETERS)
            displacements.append(dist)
            
        return coms, displacements

    def calculate_directionality(self, coords: List[Tuple[float, float]]) -> Dict:
        """
        Calculate mean movement direction from coordinate sequence.

        Parameters
        ----------
        coords : List[Tuple[float, float]]
            Sequence of (longitude, latitude) coordinates

        Returns
        -------
        dict
            Contains:
            - mean_delta_lon: Mean longitudinal displacement
            - mean_delta_lat: Mean latitudinal displacement  
            - mean_angle: Mean movement angle (degrees)
            - direction: Direction
            - movement_count: Number of movements analyzed
        """
        if len(coords) < 2:
            return {"error": "2 coordinates needed"}
        
        # Movement vectors
        delta_lons = []
        delta_lats = []
        for i in range(len(coords)-1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i+1]
            delta_lons.append(lon2 - lon1)
            delta_lats.append(lat2 - lat1)
        
        # Mean direction
        mean_delta_lon = np.mean(delta_lons)
        mean_delta_lat = np.mean(delta_lats)
        mean_angle = degrees(atan2(mean_delta_lat, mean_delta_lon)) % 360
        
        if -45 <= mean_angle < 45 or 315 <= mean_angle < 360:
            direction = "eastward (zonal-dominated)"
        elif 45 <= mean_angle < 135:
            direction = "northward (meridional-dominated)" 
        elif 135 <= mean_angle < 225:
            direction = "westward (zonal-dominated)"
        else:
            direction = "southward (meridional-dominated)"
        
        return {
            "mean_delta_lon": mean_delta_lon,
            "mean_delta_lat": mean_delta_lat,
            "mean_angle": mean_angle,
            "direction": direction,
            "movement_count": len(delta_lons)
        }