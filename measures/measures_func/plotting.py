import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing import List, Tuple

def plot_displacement(
    coordinate_list: List[Tuple[float, float]], 
    intensity_array: xr.DataArray,
    cmap: str = 'Greys',
    quiver_color: str = 'Orange',
    text_color: str = 'cyan',
    marker_color: str = 'Red',
    figsize: Tuple[int, int] = (12, 6),
    vmin: float = 0,
    vmax: float = 5,
    quiver_width: float = 0.005,
    text_fontsize: int = 15,
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
    # Sum the intensity array over all timesteps
    summed_over_all_timesteps = intensity_array.sum(axis=0)
    
    # Replace zeros with NaN for better visualization
    summed_over_all_timesteps = xr.where(summed_over_all_timesteps == 0., np.nan, summed_over_all_timesteps)
    
    # Extract latitude and longitude values from the coordinate list
    y_val_cent, x_val_cent = zip(*coordinate_list)
    
    # Calculate displacement vectors
    dx = [j - i for i, j in zip(x_val_cent[:-1], x_val_cent[1:])]
    dy = [j - i for i, j in zip(y_val_cent[:-1], y_val_cent[1:])]
    
    # Set up the plot
    plt.figure(figsize=figsize)
    
    # Plot the intensity map
    summed_over_all_timesteps.plot.imshow(cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Plot the displacement vectors (quiver plot)
    plt.quiver(
        x_val_cent[:-1], y_val_cent[:-1], dx, dy, 
        width=quiver_width, color=quiver_color, 
        angles='xy', scale_units='xy', scale=1
    )
    
    # Add text labels and markers for each centroid
    for i, (x, y) in enumerate(zip(x_val_cent, y_val_cent)):
        plt.text(x+1, y+1, str(i), fontsize=text_fontsize, c=text_color, ha='center', va='center')
        plt.scatter(x, y, c=marker_color, edgecolor='black', zorder=5)
    
    # Highlight the first and last centroids
    plt.scatter(x_val_cent[0], y_val_cent[0], c=marker_color, edgecolor='black', zorder=5, label='Start')
    plt.scatter(x_val_cent[-1], y_val_cent[-1], c=marker_color, edgecolor='black', zorder=5, label='End')
    
    # Add a legend
    plt.legend(loc='upper right')
    
    # Add labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Centroid Displacement Over Time')
    
    # Show the plot
    plt.show()