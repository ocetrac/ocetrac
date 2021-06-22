from ocetrac.model import Tracker

import pytest
import xarray as xr
import numpy as np
import scipy.ndimage
from skimage.measure import regionprops
from skimage.measure import label as label_np
import dask.array as dsa
from xarray.tests import raise_if_dask_computes


@pytest.fixture
def example_data():

    x0 = [180, 225, 360, 80, 1, 360, 1]
    y0 = [0, 20, -50, 40, -50, 40, 40]
    sigma0 = [15, 25, 30, 10, 30, 15, 10]

    lon = np.arange(0, 360) + 0.5
    lat = np.arange(-90, 90) + 0.5
    x, y = np.meshgrid(lon, lat)
    xdim = 'lon'; ydim = 'lat'

    def make_blobs(x0, y0, sigma0):
        blob = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma0 ** 2))
        return blob

    features = {}
    for i in range(len(x0)):
        features[i] = make_blobs(x0[i], y0[i], sigma0[i])

    first_image = features[0] + features[1] + features[3] - 0.5

    da = xr.DataArray(
        first_image[np.newaxis, :, :],
        dims=["time", ydim, xdim],
        coords={"time": [1], ydim: lat, xdim: lon},
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

    mask = xr.DataArray(np.ones(Anom[0, :, :].shape), coords=Anom[0, :, :].coords)
    mask[60:90, 120:190] = 0

    return Anom, mask


@pytest.mark.parametrize("radius", [8, 10])
@pytest.mark.parametrize("min_size_quartile", [0.75, 0.80])
@pytest.mark.parametrize('xdim', ['lon', 'longitude', 'whoo'])
@pytest.mark.parametrize('ydim', ['lat', 'latitude', 'whaa'])
@pytest.mark.parametrize('dask', [True, False])
def test_track(example_data, radius, min_size_quartile, xdim, ydim, dask):

    Anom, mask = example_data
    Anom = Anom.rename({'lon':xdim, 'lat':ydim})
    mask = mask.rename({'lon':xdim, 'lat':ydim})
    if dask:
        Anom = Anom.chunk({'time':1})

    tracker = Tracker(Anom, mask, radius, min_size_quartile, xdim=xdim, ydim=ydim)
    new_labels = tracker.track()

    assert (
        new_labels.attrs["percent area reject"]
        + new_labels.attrs["percent area accept"]
    ) == 1.0
    # TODO: Can we come up with an even simpler case, where we can manually write out the labels we expect?


def test_track_dask_lazy(example_data, radius=8, min_size_quartile=0.75, xdim='lon', ydim='lat'):
    # Check that dask computation is lazy and results are the same for eager/lazy compute
    Anom, mask = example_data

    Anom_chunked = Anom.chunk({'time':1})
    # mask_chunked = mask.chunk({xdim:-1, ydim:-1})
    mask_chunked = mask # dont chunk for now to not trigger the error in `self.mask == 0).all():`

    tracker = Tracker(Anom, mask, radius, min_size_quartile, xdim=xdim, ydim=ydim)
    labels = tracker.track()

    with raise_if_dask_computes(): #temporarily deactivated
        tracker_dask = Tracker(Anom_chunked, mask_chunked, radius, min_size_quartile, xdim=xdim, ydim=ydim)
        labels_dask = tracker_dask.track()

    xr.testing.assert_allclose(labels, labels_dask)
    


def test_morphological_operations(example_data, radius=8, min_size_quartile=0.75, xdim='lat', ydim='lon'):

    Anom, mask = example_data
    tracker = Tracker(Anom.chunk({'time': 1}), mask, radius, min_size_quartile, xdim, ydim)
    binary_images = tracker._morphological_operations()

    assert isinstance(binary_images.data, dsa.Array)

    ocetrac_guess = Anom.where(binary_images==True, drop=False, other=np.nan)
    best_guess = Anom.where(Anom>0, drop=False, other=np.nan)
    part = ocetrac_guess.isin(best_guess)
    whole = best_guess.isin(best_guess)

    print(part.sum().values/whole.sum().values*10)

    assert part.sum().values/whole.sum().values*100 >= 80
    assert part.sum().values == 26122

def test_apply_mask(example_data, radius=8, min_size_quartile=0.75, xdim='lat', ydim='lon'):
    
    Anom, mask = example_data
    tracker = Tracker(Anom.chunk({'time': 1}), mask, radius, min_size_quartile, xdim, ydim)
    binary_images = tracker._morphological_operations()
    binary_images_with_mask = tracker._apply_mask(binary_images, mask)
    assert (binary_images_with_mask.where(mask==0, drop=True)==0).all()

def test_filter_area(example_data, radius=8, min_size_quartile=0.75, xdim='lat', ydim='lon'):
    
    Anom, mask = example_data
    tracker = Tracker(Anom.chunk({'time': 1}), mask, radius, min_size_quartile, xdim, ydim)
    binary_images = tracker._morphological_operations()
    binary_images_with_mask = tracker._apply_mask(binary_images, mask)
    area, min_area, binary_labels, N_initial = tracker._filter_area(binary_images_with_mask)

    assert N_initial.astype(int) == 15
    assert min_area == 2761.5
    

def test_label_either(example_data, radius=8, min_size_quartile=0.75, xdim='lat', ydim='lon'):
    
    Anom, mask = example_data
    tracker = Tracker(Anom.chunk({'time': 1}), mask, radius, min_size_quartile, xdim, ydim)
    binary_images = tracker._morphological_operations()
    binary_images_with_mask = tracker._apply_mask(binary_images, mask)
    labels, num = tracker._label_either(binary_images_with_mask, return_num= True, connectivity=3)

    assert labels[2,:,:].max() == 6.
    assert all([i in labels[2,:,:] for i in range(0,6)])

def test_wrap(example_data, radius=8, min_size_quartile=0.75, xdim='lat', ydim='lon'):
    
    Anom, mask = example_data
    tracker = Tracker(Anom.chunk({'time': 1}), mask, radius, min_size_quartile, xdim, ydim)
    binary_images = tracker._morphological_operations()
    binary_images_with_mask = tracker._apply_mask(binary_images, mask)
    area, min_area, binary_labels, N_initial = tracker._filter_area(binary_images_with_mask)
    labels, num = tracker._label_either(binary_images_with_mask, return_num= True, connectivity=3)
    labels_wrapped, N_final = tracker._wrap(labels)

    assert N_final == 6
