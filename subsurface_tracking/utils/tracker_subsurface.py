import numpy as np
import xarray as xr
from scipy import ndimage
import dask.array as dsa
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

class OceanFeatureTracker:
    def __init__(self, da, radius=3, positive=True):
        """Initialize tracker with parameters
        
        Args:
            da (xarray.DataArray): Input data array (4D: time, depth, y, x)
            radius (int): Size of morphological operations
            positive (bool): Whether to track positive (True) or negative (False) features
        """
        self.da = da                # Input 4D xarray.DataArray (time, depth, y, x)
        self.radius = radius        # Radius for morphological structuring element
        self.positive = positive    # Whether to track positive or negative features
        
    def make_struct(self, connect_xy='full', connect_z=True, connect_time='none'):
        """Create 4D connectivity structure"""
        struct = np.zeros((3,3,3,3), dtype=bool) # 4D array: (time, depth, y, x)
        struct[1,1,1,1] = True # Always include the center point
        
        # --- XY Connectivity ---
        if connect_xy == 'limited':  # 4-connected
            # 4-connectivity in horizontal (left/right and up/down only)
            struct[1,1,1,0] = struct[1,1,1,2] = True  # X axis
            struct[1,1,0,1] = struct[1,1,2,1] = True  # Y axis
        elif connect_xy == 'full':  
            # 8-connectivity in horizontal (all neighbors in XY plane)
            struct[1,1,:,:] = True  # Full XY plane
        
        # --- Z Connectivity (depth) ---
        if connect_z:
            # Connect vertically in Z (depth axis)
            struct[1,:,1,1] = True 
        return struct

    def _morphological_operations(self):
        """Clean binary features with morphological ops"""
        # Convert data to binary: 1 if >0 (or <0 if negative tracking), else 0
        binary = xr.where(self.da > 0 if self.positive else self.da < 0, 1, 0)

        # Create circular structuring element
        y, x = np.ogrid[-self.radius:self.radius+1, -self.radius:self.radius+1]
        se = x**2 + y**2 <= self.radius**2

        # Process each 2D slice
        def process_slice(slice_2d):
            padded = np.pad(slice_2d, self.radius, mode='wrap') # wrap around edges
            closed = ndimage.binary_closing(padded, structure=se) # fill small holes 
            opened = ndimage.binary_opening(closed, structure=se) # remove small blobs
            return opened[self.radius:-self.radius, self.radius:-self.radius]

        # Apply to all slices
        if isinstance(binary.data, dsa.Array): # Dask array: compute in memory
            binary = binary.compute()

        # Iterate over time and depth, apply cleaning slice-by-slice
        cleaned = np.stack([
            np.stack([process_slice(binary[t,z].values) 
            for z in range(binary.shape[1])])
            for t in range(binary.shape[0])
        ])
        
        return xr.DataArray(cleaned, dims=self.da.dims, coords=self.da.coords)
    
    def track_features(self):
        """Complete tracking workflow"""
        # Step 1: Clean features using morphological operations
        binary = self._morphological_operations()

        # Step 2: Create connectivity structure
        struct = self.make_struct()

        # Step 3: Label connected features using 4D connectivity
        labels, _ = ndimage.label(binary.values, structure=struct)
        
        return xr.DataArray(
            labels,
            dims=self.da.dims,
            coords=self.da.coords,
            name='feature_labels'
        ).where(labels != 0, np.nan) # Replace 0s with NaN for clarity

def simple_temporal_connection(labeled_3d):
    """Connect features across time with minimal indexing"""
    arr = labeled_3d.values
    
    # Initialize output
    connected = np.full_like(arr, np.nan)
    next_label = 1
    label_map = {}
    
    for t in range(arr.shape[0]):
        if t == 0:
            # First timestep - keep original labels
            unique_labels = np.unique(arr[0])
            unique_labels = unique_labels[~np.isnan(unique_labels)]
            
            for lbl in unique_labels:
                mask = arr[0] == lbl
                connected[0][mask] = next_label
                label_map[(0, lbl)] = next_label
                next_label += 1
        else:
            current = arr[t]
            prev = connected[t-1]
            unique_labels = np.unique(current)
            unique_labels = unique_labels[~np.isnan(unique_labels)]
            
            for lbl in unique_labels:
                mask = current == lbl
                overlapping = prev[mask]
                overlapping_labels = overlapping[~np.isnan(overlapping)]
                
                if len(overlapping_labels) > 0:
                    # Use most common overlapping label
                    best_match = np.argmax(np.bincount(overlapping_labels.astype(int)))
                    connected[t][mask] = best_match
                else:
                    # New feature
                    connected[t][mask] = next_label
                    next_label += 1

    # Convert back to xarray with original coordinates
    return xr.DataArray(
        connected,
        dims=labeled_3d.dims,
        coords=labeled_3d.coords,
        name='connected_labels'
    )
## VISUALIZATION

import os
import imageio.v2 as imageio
import glob

def plot_3d_labeled_feature(label_da, feature_id, time_idx=0, threshold=0.5, sigma=0):
    """
    Extract and plot a labeled 3D feature at a given time index.
    """
    # Extract 3D data at the time index
    label_3d = label_da.isel(time=time_idx).values
    mask = np.nan_to_num((label_3d == feature_id).astype(float))  # 1s for feature, 0 elsewhere

    if mask.sum() == 0:
        print(f"Feature {feature_id} not found at time index {time_idx}.")
        return

    # Optional smoothing
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=sigma)

    # Extract surface
    verts, faces, _, _ = measure.marching_cubes(mask, level=threshold)

    # Get real coordinates from xarray
    z_vals = label_da.z_t.values
    y_vals = label_da.nlat.values
    x_vals = label_da.nlon.values

    dz = (z_vals[-1] - z_vals[0]) / (len(z_vals) - 1)
    dy = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)
    dx = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)

    z0, y0, x0 = z_vals[0], y_vals[0], x_vals[0]
    verts_phys = np.zeros_like(verts)
    verts_phys[:, 0] = z0 + verts[:, 0] * dz
    verts_phys[:, 1] = y0 + verts[:, 1] * dy
    verts_phys[:, 2] = x0 + verts[:, 2] * dx

    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(
        verts_phys[:, 2],  # longitude (x)
        verts_phys[:, 1],  # latitude (y)

            faces,
        verts_phys[:, 0],  # depth (z)
        cmap='plasma',
        edgecolor='none',
        alpha=0.8
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_zlabel("Depth (m)")
    ax.set_title(f"Feature {feature_id} at Time {time_idx}")
    ax.set_ylim(0, 328)
    ax.set_xlim(0, 320)
    ax.set_zlim(0, 28548.36523438)
    ax.invert_zaxis()
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6)
    cbar.set_label("Depth (m)")
    plt.tight_layout()
    return fig  # Return figure for GIF creation

def save_3d_feature_frames(label_da, feature_id, time_range, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    for time_idx in time_range:
        fig = plot_3d_labeled_feature(label_da, feature_id, time_idx)
        fig.savefig(f"{output_dir}/frame_{time_idx:03d}.png", dpi=100)
        plt.close(fig)
