import numpy as np
import xarray as xr
from functools import wraps
from scipy.interpolate import interp1d
from skimage.measure import find_contours
from haversine import haversine, Unit
from skimage.morphology import convex_hull_image
from typing import List, Tuple
import functools
import time

def log_execution_time(toggle_attr='use_decorators'):
    """Decorator to log execution time, which can be toggled on/off using a class attribute.

    Parameters
    ----------
    toggle_attr : str
        The attribute of the class that controls whether the decorator is active.
    Returns
    -------
    decorator : function
        A decorator that wraps the function to log its execution time.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, toggle_attr, True):  # Check if decorators are enabled
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
                return result
            else:
                return func(self, *args, **kwargs)  # Run function normally if disabled
        return wrapper
    return decorator


class ShapeMeasures:
    def __init__(self, lat_resolution: float = 110.574, lon_resolution: float = 111.320, use_decorators: bool = True):
        """
        Initializes the ShapeMeasures class with latitude and longitude resolutions and decorater usage.

        Parameters
        ----------
        lat_resolution : float
            Resolution in kilometers for latitude (default is 110.574 km)
        lon_resolution : float
            Resolution in kilometers for longitude (default is 111.320 km)
        use_decorators : bool
            If True, decorators will be used to log execution time (default is True)
        """
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.use_decorators = use_decorators  # Control decorator execution

    @log_execution_time()
    def calculate_area(self, lats: List[float], lons: List[float]) -> float:
        """Calculates area in square kilometers.
        
        Parameters
        ----------
        lats : List[float]
            List of latitudes in degrees.
        lons : List[float]
            List of longitudes in degrees.

        Returns
        -------
        float
            Area in square kilometers.
        """
        y, x = np.array(lats), np.array(lons)
        dlon = np.cos(np.radians(y)) * self.lon_resolution
        dlat = self.lat_resolution * np.ones(len(dlon))
        return np.sum(dlon * dlat)

    @log_execution_time()
    def calc_spatial_extents(self, one_obj: xr.Dataset) -> dict:
        """Calculates spatial extents and summary statistics for event.

        Parameters
        ----------
        one_obj : xr.Dataset
            Dataset containing labels with 'lat' and 'lon' dimensions.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'coords_full': List of coordinates for each time step.
            - 'spatial_extents': List of spatial extents for each time step.
            - 'max_spatial_extent': Maximum spatial extent across all time steps.
            - 'mean_spatial_extent': Mean spatial extent across all time steps.
            - 'cumulative_spatial_extent': Cumulative spatial extent across all time steps.
        """
        spatial_extents = []
        coords_full = []

        for i in range(len(one_obj.labels.time)):
            stacked = one_obj.labels[i, :, :].stack(zipcoords=['lat', 'lon'])
            intermed = stacked.dropna(dim='zipcoords').zipcoords.values

            if len(intermed) == 0:
                coords_full.append([])
                spatial_extents.append(0.0)
                continue

            lats, lons = zip(*intermed)
            coords = list(zip(lats, lons))
            coords_full.append(coords)
            spatial_extents.append(self.calculate_area(lats, lons))

        return {
            'coords_full': coords_full,
            'spatial_extents': spatial_extents,
            'max_spatial_extent': max(spatial_extents, default=0.0),
            'mean_spatial_extent': np.mean(spatial_extents) if spatial_extents else 0.0,
            'cumulative_spatial_extent': np.sum(spatial_extents) if spatial_extents else 0.0,
        }

    @log_execution_time()
    def calc_perimeter(self, one_obj: xr.Dataset) -> List[float]:
        """Calculates the perimeter of objects using contour detection.
        
        Parameters
        ----------
        one_obj : xr.Dataset
            Dataset containing labels with 'lat' and 'lon' dimensions.
        Returns
        -------
        List[float]
            List of perimeters for each time step in kilometers.
        """
        long_range = interp1d([0, 360], [-180, 180]) # Convert longitude from [0, 360] to [-180, 180]
        binary_mask = one_obj.labels.fillna(0).data # Replace NaN values with 0

        lat_values = one_obj.lat.values
        lon_values = one_obj.lon.values

        perimeters = []
        for i in range(len(one_obj.labels.time)):
            contours = find_contours(binary_mask[i], level=0.5) # Find contours in the binary mask
            distances = []

            for contour in contours:
                lats = lat_values[contour[:, 0].astype(int)]
                lons = lon_values[contour[:, 1].astype(int)]
                coords = list(zip(lats, long_range(lons)))

                for ind in range(len(coords) - 1):
                    distances.append(haversine(coords[ind], coords[ind + 1], Unit.KILOMETERS))
                distances.append(haversine(coords[-1], coords[0], Unit.KILOMETERS))
            perimeters.append(np.sum(distances))
        return perimeters

    @log_execution_time()
    def calc_complement_to_deformation(self, coords_full: List[List[Tuple[float, float]]], spatial_extents: List[float]) -> np.ndarray:
        """Calculates complement to deformation ratio for consecutive timesteps.
        
        Parameters
        ----------
        coords_full : List[List[Tuple[float, float]]]
            List of coordinates for each time step, where each coordinate is a tuple of (latitude, longitude).
        spatial_extents : List[float]
            List of spatial extents for each time step in square kilometers.
        
        Returns
        -------
        np.ndarray
            Array of shared area ratios for consecutive time steps.
        """
        shared_area_ratios = []
        for i in range(len(coords_full) - 1):
            shared_coords = set(coords_full[i]) & set(coords_full[i + 1])
            if shared_coords:
                y, x = zip(*shared_coords)
                shared_area = np.sum(np.cos(np.radians(y)) * self.lon_resolution * self.lat_resolution)
                shared_area_ratios.append(shared_area / (spatial_extents[i] + spatial_extents[i + 1]))
            else:
                shared_area_ratios.append(0.0)
        return np.array(shared_area_ratios)

    @log_execution_time()
    def calc_deformation(self, shared_area_ratios: List[float]) -> np.ndarray:
        """Calculates deformation as 1 - shared area ratio.

        Parameters
        ----------
        shared_area_ratios : List[float]
            List of shared area ratios for consecutive time steps.

        Returns
        -------
        np.ndarray
            Array of deformation values for consecutive time steps.
        """
        return 1 - np.array(shared_area_ratios)

    @log_execution_time()
    def calc_ratio_convexhullarea_vs_area(self, one_obj: xr.Dataset) -> List[float]:
        """Calculates the ratio of object area to convex hull area.

        Parameters
        ----------
        one_obj : xr.Dataset
            Dataset containing labels with 'lat' and 'lon' dimensions.
        
        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing:
            - List of convex hull areas for each time step.
            - List of ratios of object area to convex hull area for each time step.
        """
        ratios = []
        convex_hull_areas = []

        binary_mask = one_obj.labels.where(one_obj.labels > 0, 0).fillna(0) # Create a binary mask where values > 0 are set to 1 and NaNs are set to 0

        for i in range(len(one_obj.labels.time)):
            one_obj_one_timestep = one_obj.labels[i, :, :]
            obj_onetimestep_stacked = one_obj_one_timestep.stack(zipcoords=['lat', 'lon'])

            intermed = obj_onetimestep_stacked.dropna(dim='zipcoords').zipcoords.values

            lats, lons = zip(*intermed)

            object_area = self.calculate_area(lats, lons)
            convex_hull = convex_hull_image(binary_mask[i, :, :])

            convex_hull_coords = np.column_stack(np.where(convex_hull))

            lats = one_obj.lat.values[convex_hull_coords[:, 0]]
            lons = one_obj.lon.values[convex_hull_coords[:, 1]]

            convex_hull_area = self.calculate_area(lats, lons)
            convex_hull_areas.append(convex_hull_area)
            if convex_hull_area == 0:
                ratios.append(0.0)  # Append 0 if convex hull area is zero
                continue
            ratio = object_area / convex_hull_area
            ratios.append(ratio)
        return convex_hull_areas, ratios

    @log_execution_time()
    def calc_circularity(self, area, perimeter):
        """Calculates circularity given area and perimeter.

        Parameters
        ----------
        area : float or list of floats
            Area of the shape(s) in square kilometers.
        perimeter : float or list of floats
            Perimeter of the shape(s) in kilometers.
        
        Returns
        -------
        float or list of floats
            Circularity value(s) calculated as 4 * pi * area / perimeter^2.
            Returns NaN if perimeter is zero to avoid division by zero.
        """
        if isinstance(area, list) and isinstance(perimeter, list):
            return [
                4 * np.pi * a / (p ** 2) if p != 0 else np.nan
                for a, p in zip(area, perimeter)
            ]
        elif isinstance(area, (int, float)) and isinstance(perimeter, (int, float)):
            return 4 * np.pi * area / (perimeter ** 2) if perimeter != 0 else np.nan
        else:
            raise ValueError("Both area and perimeter should be numbers or lists of numbers of the same length.")