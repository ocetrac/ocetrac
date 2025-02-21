import pytest
import numpy as np
import xarray as xr
from measures_func.shape_measures import ShapeMeasures

# Fixture to load the real dataset
@pytest.fixture(scope="module")
def real_dataset():
    """Fixture to load the real dataset and filter for a specific object_id."""
    blobs = xr.open_dataset('0_mhwobj_rm30_rad3_0.nc')
    object_id = 1.0

    # Filter the dataset for the specific object_id
    object_count_per_time = (blobs == object_id).sum(dim=['lat', 'lon'])
    true_time_steps = object_count_per_time.time.where(object_count_per_time > 0, drop=True)
    one_obj = blobs.sel(time=true_time_steps.time)

    return one_obj

# Fixture to create a ShapeMeasures instance
@pytest.fixture
def shape_measures():
    """Fixture to create a ShapeMeasures instance."""
    return ShapeMeasures(lat_resolution=110.574, lon_resolution=111.320, use_decorators=False)

# Fixture to compute spatial extents
@pytest.fixture
def spatial_extents_data(shape_measures, real_dataset):
    """Fixture to compute spatial extents once and reuse in other tests."""
    result = shape_measures.calculate_spatial_extents(real_dataset)
    return result  # Returns the full dictionary with 'coords_full' and 'spatial_extents'

# Fixture to compute perimeter data
@pytest.fixture
def perimeter_data(shape_measures, real_dataset):
    """Fixture to compute perimeter data once and reuse in other tests."""
    return shape_measures.calculate_perimeter(real_dataset)

# Fixture to compute complement to deformation data
@pytest.fixture
def complement_to_deformation_data(shape_measures, spatial_extents_data):
    """Fixture to compute complement to deformation once and reuse in other tests."""
    coords_full = spatial_extents_data['coords_full']
    spatial_extents = spatial_extents_data['spatial_extents']
    return shape_measures.calc_complement_to_deformation(coords_full, spatial_extents)

# Test function for calculate_spatial_extents
def test_calc_spatial_extents(shape_measures, real_dataset):
    """Test the calculate_spatial_extents method with expected values."""
    # Define expected spatial extents (precomputed or known values)
    expected_spatial_extents = [8977012.402890965, 5315969.780679241]

    # Call the method under test
    result = shape_measures.calculate_spatial_extents(real_dataset)

    # Check if the output is a dictionary
    assert isinstance(result, dict), "Output is not a dictionary"

    # Check if the required keys are present
    required_keys = ['coords_full', 'spatial_extents', 'max_spatial_extent',
                     'max_spatial_extent_time', 'mean_spatial_extent', 'cumulative_spatial_extent']
    for key in required_keys:
        assert key in result, f"Key '{key}' is missing in the output"

    # Check if spatial_extents is a list and has the correct length
    assert isinstance(result['spatial_extents'], list), "spatial_extents is not a list"
    assert len(result['spatial_extents']) == len(expected_spatial_extents), "Spatial extents list length mismatch"

    # Compare each spatial extent value with the expected values
    for ext, expected in zip(result['spatial_extents'], expected_spatial_extents):
        assert np.isclose(ext, expected), f"Expected {expected}, got {ext}"

# Test function for calculate_perimeter
def test_calc_perimeter(shape_measures, perimeter_data):
    """Test the calculate_perimeter method with the dataset."""
    expected_perimeters = [21419.764799463806, 13712.718192832881]
    assert perimeter_data == expected_perimeters, f"Expected {expected_perimeters}, got {perimeter_data}"

# Test function for calc_complement_to_deformation
def test_calc_complement_to_deformation(shape_measures, complement_to_deformation_data):
    """Test the calc_complement_to_deformation method."""
    # Assert that the output is a NumPy array and has the expected length
    assert isinstance(complement_to_deformation_data, np.ndarray), "Output should be a NumPy array"

    # Check values if expected results are known
    expected_values = np.array([0.29106818])
    assert np.allclose(complement_to_deformation_data, expected_values), f"Expected {expected_values}, got {complement_to_deformation_data}"

# Test function for calc_deformation
def test_calc_deformation(shape_measures, complement_to_deformation_data):
    """Test the calc_deformation method with the output of calc_complement_to_deformation."""
    # Call the function under test
    deformation = shape_measures.calc_deformation(complement_to_deformation_data)

    # Assert that the output is a NumPy array
    assert isinstance(deformation, np.ndarray), "Output should be a NumPy array"

    # Check values if expected results are known
    expected_values = np.array([0.70893182])
    assert np.allclose(deformation, expected_values), f"Expected {expected_values}, got {deformation}"

# Test function for calc_ratio_convexhullarea_vs_area
def test_calc_ratio_convexhullarea_vs_area(shape_measures, real_dataset):
    """Test the calc_ratio_convexhullarea_vs_area method."""
    convex_hull_areas, ratios = shape_measures.calc_ratio_convexhullarea_vs_area(real_dataset)

    expected_convex_hull_areas = [14328360.993213397, 5761565.886890195]
    expected_ratios = [0.6265205355408697, 0.9226605900272946]

    # Assert that the outputs match expected values
    assert np.allclose(convex_hull_areas, expected_convex_hull_areas), (
        f"Expected {expected_convex_hull_areas}, but got {convex_hull_areas}"
    )
    assert np.allclose(ratios, expected_ratios), (
        f"Expected {expected_ratios}, but got {ratios}"
    )

# Test function for calc_circularity
def test_calc_circularity(shape_measures, spatial_extents_data, perimeter_data):
    """Test the calc_circularity method."""
    spatial_extents = spatial_extents_data['spatial_extents']
    circularity = shape_measures.calc_circularity(spatial_extents, perimeter_data)

    expected_circularity = [0.24587382289134527, 0.35525914436995565]
    assert np.allclose(circularity, expected_circularity), (
        f"Expected {expected_circularity}, but got {circularity}"
    )