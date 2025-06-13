"""
motion_measures.py

Class for analyzing motion characteristics of geospatial objects:
- Centroid tracking
- Center of mass calculations
- Displacement, velocity and directionality metrics
"""

import warnings

from math import atan2, degrees
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from haversine import Unit, haversine
from scipy import ndimage
from scipy.interpolate import interp1d
from skimage.measure import label as label_np
from skimage.measure import regionprops


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

    def _calculate_com(
        self, intensity_image: xr.DataArray
    ) -> List[Tuple[float, float]]:
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
        - Warns if objects in the intensity image span all longitudes.
        """
        img = intensity_image.fillna(0)
        img_by_lon = img.sum("lat")[::-1]
        shift = img_by_lon.argmin("lon").values
        if img_by_lon[shift] > 0:
            warnings.warn(
                "Objects span all longitudes, centroid calculations may be incorrect."
            )

        rolled = img.roll(lon=shift, roll_coords=True)
        com = ndimage.center_of_mass(rolled.data)
        centroid = (
            float(rolled.lat[round(com[0])].values),
            float(rolled.lon[round(com[1])].values),
        )
        return [centroid]

    def _label_regions(self, binary_data: np.ndarray) -> np.ndarray:
        """Label connected regions in binary data. Imposes periodic boundary and wraps labels.

        Parameters
        ----------
        binary_data : np.ndarray
            Binary data array with dimensions (lat, lon)

        Returns
        -------
        np.ndarray
            Labeled regions with unique integer labels for each region, with periodic boundary conditions
        """
        binary_data = np.nan_to_num(binary_data, nan=0)
        binary_data = np.where(np.isfinite(binary_data), binary_data, 0)
        labels = label_np(binary_data, background=0)

        # Impose periodic boundary and wrap labels
        first_column = labels[..., 0]
        last_column = labels[..., -1]
        unique_first = np.unique(first_column[first_column > 0])
        for i in enumerate(unique_first):
            first = np.where(first_column == i[1])
            last = last_column[first[0]]
            bad_labels = np.unique(last[last > 0])
            replace = np.isin(labels, bad_labels)
            labels[replace] = i[1]

        labels_wrapped = np.unique(labels, return_inverse=True)[1].reshape(labels.shape)
        return labels_wrapped

    def calculate_centroids(
        self, labels: xr.DataArray, timestep: int
    ) -> List[Tuple[float, float]]:
        """
        Calculate centroids for all regions at a specified timestep.

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
        centroids = []
        timestep_data = labels.isel(time=timestep)
        labeled = self._label_regions(timestep_data.values)
        labeled = xr.DataArray(
            labeled, dims=timestep_data.dims, coords=timestep_data.coords
        )
        labeled = labeled.where(timestep_data > 0, other=np.nan)
        for label in np.unique(labeled)[:-1]:
            centroids.append(
                self._calculate_com(xr.where(labeled == label, 1, np.nan))[0]
            )

        return centroids

    def calculate_coms(
        self, labels: xr.DataArray, intensity: xr.DataArray, timestep: int
    ) -> List[Tuple[float, float]]:
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
        coms = []
        timestep_data = labels.isel(time=timestep)
        timestep_intensity = intensity.isel(time=timestep)
        labeled = self._label_regions(timestep_data.values)
        labeled = xr.DataArray(
            labeled, dims=timestep_data.dims, coords=timestep_data.coords
        )
        labeled = labeled.where(timestep_data > 0, other=np.nan)
        for label in np.unique(labeled)[:-1]:
            coms.append(
                self._calculate_com(timestep_intensity.where(labeled == label, np.nan))[
                    0
                ]
            )

        return coms

    def calculate_centroid_displacement(
        self, labels: xr.DataArray
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
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
            dist = haversine(
                centroids[i], centroids[i + 1], Unit.KILOMETERS, normalize=True
            )
            displacements.append(dist)

        return centroids, displacements

    def calculate_com_displacement(
        self, intensity: xr.DataArray
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
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
            dist = haversine(coms[i], coms[i + 1], Unit.KILOMETERS, normalize=True)
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
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
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
            "movement_count": len(delta_lons),
        }
