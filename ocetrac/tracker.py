"""
This module applies all Ocetrac methods for labeling objects in 3 dimensions.
"""
import dask.array as dsa
import numpy as np
import scipy.ndimage
import xarray as xr

from skimage.measure import regionprops_table 
from dask.base import is_dask_collection
from dask import persist
from dask_image.ndmeasure import label as label_dask


def _apply_mask(binary_images, mask):
    binary_images_with_mask = binary_images.where(mask == 1, drop=False, other=0)
    return binary_images_with_mask


class Tracker:
    """
    Tracker object for applying Ocetrac
    """

    def __init__(
        self, da, mask, radius, min_size_quartile, timedim, xdim, ydim, positive=True
    ):
        self.da = da
        self.mask = mask
        self.radius = radius
        self.min_size_quartile = min_size_quartile
        self.timedim = timedim
        self.xdim = xdim
        self.ydim = ydim
        self.positive = positive

        if (timedim, ydim, xdim) != da.dims:
            try:
                da = da.transpose(timedim, ydim, xdim)
            except:
                raise ValueError(
                    f"Ocetrac currently only supports 3D DataArrays. The dimensions should only contain ({timedim}, {xdim}, and {ydim}). Found {list(da.dims)}"
                )

        if not is_dask_collection(da.data):
            raise ValueError('The input DataArray is not backed by a Dask array. Please chunk (in time), and try again.  :)')

    def track(self):
        """
        Label and track image features.

        Parameters
        ----------
        da : xarray.DataArray
            The data to label.

        mask : xarray.DataArray
            The mask of points to ignore. Must be binary where 1 = true point and 0 = background to be ignored.

        radius : int
            The size of the structuring element used in morphological opening and closing. Radius specified by the number of grid units.
            
            Structuring elements are defined such that cells are included if their distance from the origin in index space is
            strictly less than the radius. For example, a radius of 1 means that the structuring element includes just an
            individual pixel (such that applying the morhological closing and opening just returns the original binary array),
            while a radius of 2 would additionally include the eight cells cells adjacent to the origin. As the radius increases,
            the shape of the structure element asymptotes to a circle centered on the origin.

        min_size_quartile : float
            The quantile used to define the threshold of the smallest area object retained in tracking. Value should be between 0 and 1.
            A value of exactly 0 means objects of any size are retained while a value of 1 retains just the single largest event.
            Higher values of `min_size_quartile` result in improved performance because less events are stored in memory and need to be
            compared.

        timedim : str
            The name of the time dimension

        xdim : str
            The name of the x dimension

        ydim : str
            The name of the y dimension

        positive : bool
            True if da values are expected to be positive, false if they are negative. Default argument is True

        Returns
        -------
        labels : xarray.DataArray
            Integer labels of the connected regions.
        """

        if (self.mask == 0).all():
            raise ValueError(
                "Found only zeros in `mask` input. The mask should indicate valid regions with values of 1"
            )

        # Convert data to binary, define structuring element, and perform morphological closing then opening
        binary_images = self._morphological_operations()

        # Apply mask
        binary_images_with_mask = _apply_mask(
            binary_images, self.mask
        )  # perhaps change to method? JB

        # Filter area
        area, min_area, binary_labels, N_initial = self._filter_area(
            binary_images_with_mask
        )

        # Label objects (using dask_label, *wraps at the same time*) -- connectivity now in time
        labels_wrapped, N_final = label_dask(binary_labels, structure=np.ones((3,3,3)), wrap_axes=(2,))
        labels_wrapped, N_final = persist(labels_wrapped, N_final) # Persist both in memory simultaneously...
        N_final = N_final.compute()
        
        # Final labels to DataArray
        new_labels = xr.DataArray(
            labels_wrapped, coords=binary_labels.coords, dims=binary_labels.dims, attrs=binary_labels.attrs
        )
        new_labels = new_labels.where(new_labels != 0, drop=False, other=np.nan)

        ## Metadata

        # Calculate Percent of total object area retained after size filtering
        sum_tot_area = int(area.sum().item())

        reject_area = area.where(area <= min_area, drop=True)
        sum_reject_area = int(reject_area.sum().item())
        percent_area_reject = sum_reject_area / sum_tot_area

        accept_area = area.where(area > min_area, drop=True)
        sum_accept_area = int(accept_area.sum().item())
        percent_area_accept = sum_accept_area / sum_tot_area

        new_labels = new_labels.rename("labels")
        new_labels.attrs["initial objects identified"] = int(N_initial)
        new_labels.attrs["final objects tracked"] = int(N_final)
        new_labels.attrs["radius"] = self.radius
        new_labels.attrs["size quantile threshold"] = self.min_size_quartile
        new_labels.attrs["min area"] = min_area
        new_labels.attrs["percent area reject"] = percent_area_reject
        new_labels.attrs["percent area accept"] = percent_area_accept

        print("initial objects identified \t", int(N_initial))
        print("final objects tracked \t", int(N_final))

        return new_labels

    ### PRIVATE METHODS - not meant to be called by user ###

    def _morphological_operations(self):
        """Converts xarray.DataArray to binary, defines structuring element, and performs morphological closing then opening.
        Parameters
        ----------
        da     : xarray.DataArray
                The data to label
        radius : int
                Length of grid spacing that defines the radius of the structuring element used in morphological closing and opening.
                
                Structuring elements are defined such that cells are included if their distance from the origin in index space is
                strictly less than the radius. For example, a radius of 1 means that the structuring element includes just an
                individual pixel (such that applying the morhological closing and opening just returns the original binary array),
                while a radius of 2 would additionally include the eight cells cells adjacent to the origin. As the radius increases,
                the shape of the structure element asymptotes to a circle centered on the origin.

        """

        # Convert images to binary. All positive values == 1, otherwise == 0
        if self.positive == True:
            bitmap_binary = self.da.where(self.da > 0, drop=False, other=0)

        elif self.positive == False:
            bitmap_binary = self.da.where(self.da < 0, drop=False, other=0)

        bitmap_binary = bitmap_binary.where(bitmap_binary == 0, drop=False, other=1)

        # Define structuring element
        diameter = self.radius * 2
        x = np.arange(-self.radius, self.radius + 1)
        x, y = np.meshgrid(x, x)
        r = x**2 + y**2
        se = r < self.radius**2

        def binary_open_close(bitmap_binary):
            bitmap_binary_padded = np.pad(
                bitmap_binary, ((diameter, diameter), (diameter, diameter)), mode="wrap"
            )
            # If the radius is equal to 1, the structuring element is just an individual pixel, so
            # it is faster to just skip the calls to `binary_closing` and `binary_opening`
            if self.radius == 1:
                s2 = bitmap_binary_padded
            elif self.radius > 1:
                s1 = scipy.ndimage.binary_closing(bitmap_binary_padded, se, iterations=1)
                s2 = scipy.ndimage.binary_opening(s1, se, iterations=1)
            else:
                raise ValueError("radius must be an integer greater than or equal to 1")
                
            unpadded = s2[diameter:-diameter, diameter:-diameter]
            return unpadded

        mo_binary = xr.apply_ufunc(
            binary_open_close,
            bitmap_binary,
            input_core_dims=[[self.ydim, self.xdim]],
            output_core_dims=[[self.ydim, self.xdim]],
            output_dtypes=[bitmap_binary.dtype],
            vectorize=True,
            dask="parallelized",
        )
        return mo_binary

    def _filter_area(self, binary_images):
        """calculate area with regionprops"""

        # Label time-independent in 2D (i.e. no time connectivity!) and wrap in xdim
        connectivity = np.zeros((3,3,3))
        connectivity[1,:,:] = 1
        labels_wrapped, N_initial = label_dask(binary_images, structure=connectivity, wrap_axes=(2,))
        labels_wrapped, N_initial = persist(labels_wrapped, N_initial) # Persist both in memory simultaneously...
        
        N_initial = N_initial.compute()
        labels_wrapped = xr.DataArray(labels_wrapped, coords=binary_images.coords, dims=binary_images.dims, attrs=binary_images.attrs)
        
        # Calculate Area of each object and keep objects larger than threshold
        def regionprops_slice(labels):
            props_slice = regionprops_table(labels.astype('int'), properties=['label', 'area'])
            return props_slice
        
        props = xr.apply_ufunc(regionprops_slice, labels_wrapped,
                input_core_dims=[[self.ydim, self.xdim]],
                output_core_dims=[[]],
                output_dtypes=[object],
                vectorize=True,
                dask='parallelized')
        
        props_da = xr.concat([xr.Dataset({key: (['label'], value) for key, value in item.items()}) for item in props.values], dim='label')
        
        labelprops = props_da['label']        
        area = props_da['area']

        if area.size == 0:
            raise ValueError(f'No objects were detected. Try changing radius or min_size_quartile parameters.')
        
        min_area = np.percentile(area, self.min_size_quartile*100)
        print(f'minimum area: {min_area}') 
        
        keep_labels = labelprops.where(area>=min_area, drop=True)
        keep_where = labels_wrapped.isin(keep_labels)
        out_labels = xr.where(~keep_where, 0, labels_wrapped)

        # Convert labels to binary. All positive values == 1, otherwise == 0
        binary_labels = out_labels.where(out_labels==0, drop=False, other=1)

        return area, min_area, binary_labels, N_initial

