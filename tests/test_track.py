from ocetrac.track import (
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
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)
    
    assert (binary_images_with_mask.where(mask==0, drop=True)==0).all()
    

def test_label_either():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)

    labels, num = _label_either(binary_images_with_mask, return_num= True, connectivity=3)
    
    assert labels[2,:,:].max() == 6.
    assert all([i in labels[2,:,:] for i in range(0,6)])
    

def test_filter_area():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)
    labels, num = _label_either(binary_images_with_mask, return_num= True, connectivity=3)
    
    min_size_quartile = .75
    area, min_area, labels_greater_minsize, labelprops = _filter_area(labels, min_size_quartile)
    
    assert min_area == 3413.0
    assert all([i in labels_greater_minsize for i in [1,4]])
    assert (area.label[area>=min_area].values == np.unique(labels_greater_minsize[labels_greater_minsize>0])).all()
    
    
def test_wrap():
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    binary_images = _morphological_operations(Anom_dask, radius=8)
    binary_images_with_mask = binary_images.where(mask==1, drop=False, other=0)
    labels, num = _label_either(binary_images_with_mask, return_num= True, connectivity=3)
    min_size_quartile = .75
    area, min_area, labels_greater_minsize, labelprops = _filter_area(labels, min_size_quartile)
    
    out_labels, N = _wrap(labels_greater_minsize)
    
    assert np.max(out_labels) == 2


@pytest.mark.parametrize("radius", [8, 10])
@pytest.mark.parametrize("area_quantile", [0.75, 0.80])
def test_track(radius, area_quantile):
    Anom, mask = example_anomaly_data()
    Anom_dask = Anom.chunk({'time': 1})
    new_labels = track(Anom, mask, radius=radius, area_quantile=area_quantile)
    
    assert (new_labels.percent_area_reject + new_labels.percent_area_accept) == 1.0

    props = regionprops(new_labels.astype('int'))
    labelprops = [p.label for p in props]
    labelprops = xr.DataArray(labelprops, dims=['label'], coords={'label': labelprops}) 
    area = xr.DataArray([p.area for p in props], dims=['label'], coords={'label': labelprops}) 
    
    assert area.min() >= new_labels.min_area
    