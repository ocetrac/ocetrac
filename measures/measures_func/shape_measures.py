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
    """Decorator to log execution time, which can be toggled on/off using a class attribute."""
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
        A class to compute shape-based metrics for geospatial objects.

        Parameters:
            lat_resolution (float): Resolution of latitude in km per degree.
            lon_resolution (float): Resolution of longitude in km per degree at the equator.
            use_decorators (bool): Toggle for using decorators like execution time logging.
        """
        self.lat_resolution = lat_resolution
        self.lon_resolution = lon_resolution
        self.use_decorators = use_decorators  # Control decorator execution

    @log_execution_time()
    def calculate_area(self, lats: List[float], lons: List[float]) -> float:
        """Computes area in square kilometers."""
        y, x = np.array(lats), np.array(lons)
        dlon = np.cos(np.radians(y)) * self.lon_resolution
        dlat = self.lat_resolution * np.ones(len(dlon))
        return np.sum(dlon * dlat)

    @log_execution_time()
    def calculate_spatial_extents(self, one_obj: xr.Dataset) -> dict:
        """Computes spatial extents and summary statistics."""
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
            'max_spatial_extent_time': np.argmax(spatial_extents) if spatial_extents else -1,
            'mean_spatial_extent': np.mean(spatial_extents) if spatial_extents else 0.0,
            'cumulative_spatial_extent': np.sum(spatial_extents) if spatial_extents else 0.0,
        }

    @log_execution_time()
    def calculate_perimeter(self, one_obj: xr.Dataset) -> List[float]:
        """Computes the perimeter of objects using contour detection."""
        # Convert longitude from [0, 360] to [-180, 180]
        long_range = interp1d([0, 360], [-180, 180])

        # Replace NaN values with 0 and extract the data
        binary_mask = one_obj.labels.fillna(0).data

        # Precompute latitude and longitude values
        lat_values = one_obj.lat.values
        lon_values = one_obj.lon.values

        # Initialize the list to store perimeters
        perimeters = []

        # Iterate over each time step
        for i in range(len(one_obj.labels.time)):
            # Find contours in the binary mask for the current time step
            contours = find_contours(binary_mask[i], level=0.5)

            # Initialize distances for the current time step
            distances = []

            # Iterate over each contour
            for contour in contours:
                # Extract latitudes and longitudes from the contour
                lats = lat_values[contour[:, 0].astype(int)]
                lons = lon_values[contour[:, 1].astype(int)]

                # Convert longitudes to the range [-180, 180]
                coords = list(zip(lats, long_range(lons)))

                # Calculate distances between consecutive points in the contour
                for ind in range(len(coords) - 1):
                    distances.append(haversine(coords[ind], coords[ind + 1], Unit.KILOMETERS))

                # Close the contour loop by connecting the last point to the first
                distances.append(haversine(coords[-1], coords[0], Unit.KILOMETERS))

            # Sum the distances to get the perimeter for the current time step
            perimeters.append(np.sum(distances))

        return perimeters

    @log_execution_time()
    def calc_complement_to_deformation(self, coords_full: List[List[Tuple[float, float]]], spatial_extents: List[float]) -> np.ndarray:
        """Computes complement to deformation ratio for consecutive timesteps."""
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
        """Computes deformation as 1 - shared area ratio."""
        return 1 - np.array(shared_area_ratios)

    @log_execution_time()
    def calc_ratio_convexhullarea_vs_area(self, one_obj: xr.Dataset) -> List[float]:
        """Computes the ratio of object area to convex hull area."""
        # Initialize the list to store the ratios
        ratios = []
        convex_hull_areas = []
        # Create a binary mask where values > 0 are set to 1 and NaNs are set to 0
        binary_mask = one_obj.labels.where(one_obj.labels > 0, 0).fillna(0)

        # Iterate over each time step
        for i in range(len(one_obj.labels.time)):
            # Extract the object labels for the current time step
            one_obj_one_timestep = one_obj.labels[i, :, :]

            # Stack the latitude and longitude dimensions into a single dimension for easier processing
            obj_onetimestep_stacked = one_obj_one_timestep.stack(zipcoords=['lat', 'lon'])

            # Drop NaN values and extract the coordinates of the object
            intermed = obj_onetimestep_stacked.dropna(dim='zipcoords').zipcoords.values

            # Unpack the latitude and longitude values from the coordinates
            lats, lons = zip(*intermed)

            # Calculate the area of the object using the helper function
            object_area = self.calculate_area(lats, lons)

            # Calculate the convex hull of the binary mask for the current time step
            convex_hull = convex_hull_image(binary_mask[i, :, :])

            # Extract the coordinates of the convex hull
            convex_hull_coords = np.column_stack(np.where(convex_hull))

            # Convert pixel coordinates to latitude and longitude
            lats = one_obj.lat.values[convex_hull_coords[:, 0]]
            lons = one_obj.lon.values[convex_hull_coords[:, 1]]

            # Calculate the area of the convex hull using the helper function
            convex_hull_area = self.calculate_area(lats, lons)
            convex_hull_areas.append(convex_hull_area)
            # Handle edge case: convex hull area is zero
            if convex_hull_area == 0:
                ratios.append(0.0)  # Append 0 if convex hull area is zero
                continue

            # Calculate the ratio of object area to convex hull area
            ratio = object_area / convex_hull_area

            # Append the ratio to the list
            ratios.append(ratio)

        return convex_hull_areas, ratios

    @log_execution_time()
    def calc_circularity(self, area, perimeter):
        """
        Calculate circularity given area and perimeter.

        Parameters:
        area (float or list of floats): The area of the object(s).
        perimeter (float or list of floats): The perimeter of the object(s).

        Returns:
        float or list of floats: Circularity value(s).
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