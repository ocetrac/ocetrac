import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize

def plot_3d_labeled_feature(
    label_da, 
    feature_id, 
    time_idx=0, 
    threshold=0.5, 
    sigma=0,
    alpha=0.7, 
    cmap='viridis', 
    add_contours=True, 
    elev=25, 
    azim=-60, 
    figsize=(10, 8)):
    """
    Extract and plot an event
    
    Parameters
    ----------
    label_da : xarray.DataArray
        4D array (time, depth, lat, lon) with object labels
    feature_id : int
        Object ID to extract and plot
    time_idx : int
        Time index to visualize
    threshold : float, default=0.5
        Marching cubes threshold (0-1)
    sigma : float, default=0
        Gaussian smoothing sigma (0 = no smoothing)
    alpha : float, default=0.7
        Surface transparency
    cmap : str, default='viridis'
        Colormap for depth/height
    add_contours : bool, default=True
        Add contour lines for better shape definition
    elev, azim : float, default=25, -60
        Viewing angle (elevation, azimuth)
    figsize : tuple, default=(10, 8)
        Figure size
    
    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    
    label_3d = label_da.isel(time=time_idx).values
    mask = (label_3d == feature_id).astype(float)
    
    mask = np.nan_to_num(mask)
    
    if mask.sum() == 0:
        print(f"Feature {feature_id} not found at time index {time_idx}.")
        return None, None
    
    # Optional smoothing
    if sigma > 0:
        mask = gaussian_filter(mask, sigma=sigma)
        print(f"Applied Gaussian smoothing (sigma={sigma})")
    
    # Extract isosurface
    try:
        verts, faces, normals, values = measure.marching_cubes(mask, level=threshold)
        print(f"Extracted surface: {len(verts)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"Marching cubes failed: {e}")
        return None, None
    
    # Get physical coordinates
    z_vals = label_da.z_t.values
    y_vals = label_da.nlat.values
    x_vals = label_da.nlon.values
    
    # Calculate scaling and origin
    z0, y0, x0 = z_vals[0], y_vals[0], x_vals[0]
    dz = (z_vals[-1] - z_vals[0]) / (len(z_vals) - 1)
    dy = (y_vals[-1] - y_vals[0]) / (len(y_vals) - 1)
    dx = (x_vals[-1] - x_vals[0]) / (len(x_vals) - 1)
    
    # Transform vertices to physical coordinates
    verts_phys = np.zeros_like(verts)
    verts_phys[:, 0] = z0 + verts[:, 0] * dz  # depth
    verts_phys[:, 1] = y0 + verts[:, 1] * dy  # latitude
    verts_phys[:, 2] = x0 + verts[:, 2] * dx  # longitude
    
    # Create figure with 3D subplot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color mapping based on depth
    depth_values = verts_phys[:, 0]
    norm = Normalize(vmin=depth_values.min(), vmax=depth_values.max())
    colors = plt.cm.get_cmap(cmap)(norm(depth_values))
    
    # Plot main surface
    surf = ax.plot_trisurf(
        verts_phys[:, 2],  # x (longitude)
        verts_phys[:, 1],  # y (latitude)
        verts_phys[:, 0],  # z (depth)
        triangles=faces,
        cmap=cmap,
        alpha=alpha,
        edgecolor='none',
        antialiased=True
    )
    
    # Add contour lines (optional)
    if add_contours and len(faces) > 100:
        # Plot wireframe contours on the surface
        ax.plot_trisurf(
            verts_phys[:, 2], verts_phys[:, 1], verts_phys[:, 0],
            triangles=faces,
            color='black',
            alpha=0.05,
            linewidth=0.1,
            edgecolor='black'
        )
    
    # Set labels with units
    ax.set_xlabel("nlon", fontsize=10, labelpad=10)
    ax.set_ylabel("nlat", fontsize=10, labelpad=10)
    ax.set_zlabel("Depth (cm)", fontsize=10, labelpad=10)
    
    # Set title
    time_val = label_da.time.values[time_idx] if 'time' in label_da.dims else time_idx
    ax.set_title(
        f"Feature {feature_id} | Time = {time_val}\nVolume: {mask.sum():.0f} voxels | Depth: {np.max(verts_phys[:, 0]):.0f}m",
        fontsize=12, fontweight='bold'
    )
    
    # Set axis limits (auto or from data)
    ax.set_xlim([x_vals.min(), x_vals.max()])
    ax.set_ylim([y_vals.min(), y_vals.max()])
    ax.set_zlim([z_vals.max(), z_vals.min()])  # Invert for depth
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Depth (cm)", fontsize=10)
    
    # Add grid for better reference
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, ax