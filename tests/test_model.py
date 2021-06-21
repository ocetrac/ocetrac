from ocetrac.model import (
    _morphological_operations,
    _apply_mask,
    _label_either,
    _filter_area,
    _wrap,
    track,
)

import pytest
import xarray as xr
import numpy as np
import scipy.ndimage
from skimage.measure import regionprops 
from skimage.measure import label as label_np
import dask.array as dsa

def example_anomaly_data():
    x0 = [180, 225, 360, 80, 1, 360, 1]
    y0 = [0, 20, -50, 40, -50, 40, 40]
    sigma0 = [15, 25, 30, 10, 30, 15, 10]

    lon = np.arange(0, 360) + 0.5
    lat = np.arange(-90, 90) + 0.5
    x, y = np.meshgrid(lon, lat)

    def make_blobs(x0, y0, sigma0):
        blob = np.exp(-((x - x0)**2 + (y - y0)**2)/(2*sigma0**2))
        return blob

    features = {}
    for i in range(len(x0)):
        features[i] = make_blobs(x0[i], y0[i], sigma0[i])

    first_image = features[0]+features[1]+features[3] -.5

    da = xr.DataArray(first_image[np.newaxis,:,:], dims=['time','lat', 'lon'],
                      coords={'time':[1],'lat': lat, 'lon': lon})

    da_shift_01 = da.shift(lon=0, lat=-20, fill_value=-.5)
    da_shift_02 = da.shift(lon=0, lat=-40, fill_value=-.5)+(features[2]+features[4]+features[5]+features[6])
    da_shift_03 = da.shift(lon=0, lat=-40, fill_value=-.5)+(features[2]+features[5]+features[6])

    Anom = xr.concat((da,
                      da_shift_01,
                      da_shift_02,
                      da_shift_03,), dim='time')
    
    Anom['time'] = np.arange(1,5)
    
    mask = xr.DataArray(np.ones(Anom[0,:,:].shape), coords=Anom[0,:,:].coords)
    mask[60:90,120:190] = 0
    
    return Anom, mask


def test_morphological_operations():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    
    assert isinstance(binary_images.data, dsa.Array)
    
    ocetrac_guess = Anom.where(binary_images==True, drop=False, other=np.nan) 
    best_guess = Anom.where(Anom>0, drop=False, other=np.nan) 
    part = ocetrac_guess.isin(best_guess)
    whole = best_guess.isin(best_guess)

    assert part.sum().values/whole.sum().values*100 >= 80
    assert part.sum().values == 26122
    
    
def test_apply_mask():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = _apply_mask(mask, binary_images)    
    assert (binary_images_with_mask.where(mask==0, drop=True)==0).all()
    
def test_filter_area():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = _apply_mask(mask, binary_images)

    min_size_quartile = .75
    area, min_area, binary_labels, N_initial = _filter_area(binary_images_with_mask, min_size_quartile)
    
    assert min_area == 3853.5
    assert N_initial.astype(int) == 12


def test_label_either():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = _apply_mask(mask, binary_images)

    labels, num = _label_either(binary_images_with_mask, return_num= True, connectivity=3)
    
    assert labels[2,:,:].max() == 6.
    assert all([i in labels[2,:,:] for i in range(0,6)])
    


    
def test_wrap():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = _apply_mask(mask, binary_images)
    min_size_quartile = .75
    
    area, min_area, binary_labels, N_initial = _filter_area(binary_images_with_mask, min_size_quartile)    
    labels, num = _label_either(binary_images_with_mask, return_num= True, connectivity=3)
    labels_wrapped, N_final = _wrap(labels)

    assert N_final == 4


@pytest.mark.parametrize("radius", [8, 10])
@pytest.mark.parametrize("min_size_quartile", [0.75, 0.80])
def test_track(radius, min_size_quartile):
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    new_labels = track(Anom, mask, radius=radius, min_size_quartile=min_size_quartile)
    assert (new_labels.attrs['percent area reject'] + new_labels.attrs['percent area accept']) == 1.0