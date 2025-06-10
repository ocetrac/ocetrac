import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def plot_displacement(
    coordinate_list: List[Tuple[float, float]], 
    intensity_array: xr.DataArray,
    cmap: str = 'viridis',
    quiver_color: str = 'Orange',
    text_color: str = 'k',
    marker_color: str = 'Red',
    figsize: Tuple[int, int] = (12, 6),
    vmin: float = 0,
    vmax: float = 5,
    quiver_width: float = 0.005,
    text_fontsize: int = 12,
) -> None:
    """
    Plots the displacement of centroids over time on an intensity map.

    Parameters
    ----------
    coordinate_list : List[Tuple[float, float]]
        List of (latitude, longitude) coordinates for centroids over time.
    intensity_array : xr.DataArray
        Intensity array to plot the displacement on. Must have dimensions (time, lat, lon).
    cmap : str, optional
        Colormap for the intensity map. Default is 'Greys'.
    quiver_color : str, optional
        Color of the quiver arrows representing displacement. Default is 'Orange'.
    text_color : str, optional
        Color of the text labels for each centroid. Default is 'Red'.
    marker_color : str, optional
        Color of the markers for each centroid. Default is 'Red'.
    figsize : Tuple[int, int], optional
        Size of the figure. Default is (12, 6).
    vmin : float, optional
        Minimum value for the intensity map colormap. Default is 0.
    vmax : float, optional
        Maximum value for the intensity map colormap. Default is 5.
    quiver_width : float, optional
        Width of the quiver arrows. Default is 0.005.
    text_fontsize : int, optional
        Font size for the text labels. Default is 15.

    Returns
    -------
    None
        Displays the plot.
    """
    plt.figure(figsize=figsize)
    
    # Plot the contourf of the first timestep
    intensity_array[:,:,:].sum(dim='time').plot.contourf(cmap=cmap, vmin=vmin, vmax=vmax)
        
    # Process and plot the centroid path
    y_val_cent, x_val_cent = zip(*coordinate_list)
    
    dx = [j - i for i, j in zip(x_val_cent[:-1], x_val_cent[1:])]
    dy = [j - i for i, j in zip(y_val_cent[:-1], y_val_cent[1:])]
    
    plt.quiver(
        x_val_cent[:-1], y_val_cent[:-1], dx, dy, 
        width=quiver_width, color=quiver_color, 
        angles='xy', scale_units='xy', scale=1
    )
    
    for i, (x, y) in enumerate(zip(x_val_cent, y_val_cent)):
        plt.text(x+1, y+1, str(i), fontsize=text_fontsize, c=text_color, ha='center', va='center', zorder=20)
        plt.scatter(x, y, c=marker_color, edgecolor='black', zorder=15)
    
    plt.scatter(x_val_cent[0], y_val_cent[0], c=marker_color, edgecolor='black', zorder=5, label='Start')
    plt.scatter(x_val_cent[-1], y_val_cent[-1], c=marker_color, edgecolor='black', zorder=5, label='End')
    plt.legend(loc='upper right')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()