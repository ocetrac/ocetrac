"""
cesm2_lens_utils.py - Utility functions for accessing CESM2-LENS model data

Functions to identify, path, and load CESM2-LENS model outputs. 
"""

import os
import xarray as xr

def get_var_paths(directory, var):
    """
    Get unique file paths for CESM2-LENS historical and future scenarios.

    Parameters
    ----------
    directory : str
        Path to CESM2-LENS NetCDF files on glade
    var : str
        Variable name (unused)

    Returns
    -------
    tuple
        A tuple containing two sorted lists:
        - path_intermed_hist : List of unique historical scenario file prefixes
        - path_intermed_fut : List of unique future scenario file prefixes

    Notes
    -----
    - Historical files with prefixes: 'b.e21.BHISTcmip6.', 'b.e21.BHISTsmbb.'
    - Future files with prefixes: 'b.e21.BSSP370cmip6.', 'b.e21.BSSP370smbb.'
    """
    prefixes_to_match_fut = ['b.e21.BSSP370cmip6.', 'b.e21.BSSP370smbb.']
    prefixes_to_match_hist = ['b.e21.BHISTcmip6.', 'b.e21.BHISTsmbb.']
        
    prefixes_fut = list()
    prefixes_hist = list()
        
    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check and add prefixes for future files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_fut) and filename.endswith('.nc'):
            prefixes_fut.append(filename.rsplit('.', 3)[0])
        # Check and add prefixes for historical files
        if any(filename.startswith(prefix) for prefix in prefixes_to_match_hist) and filename.endswith('.nc'):
            prefixes_hist.append(filename.rsplit('.', 3)[0])
        
    # Convert to sets to remove duplicates, then back to sorted lists
    prefixes_hist_set = set(prefixes_hist)
    sorted_unique_list_hist = sorted(prefixes_hist_set)

    prefixes_fut_set = set(prefixes_fut)
    sorted_unique_list_fut = sorted(prefixes_fut_set)

    path_intermed_fut = sorted_unique_list_fut
    path_intermed_hist = sorted_unique_list_hist

    return path_intermed_hist, path_intermed_fut

def get_ds_var(directory, var, comp, index_hist):
    """
    Load historical and future datasets for a specified variable and component

    Parameters
    ----------
    directory : str
        Path to the directory containing CESM LENS2 NetCDF files
    var : str
        Variable name to load (e.g., 'tas', 'pr')
    comp : str
        Model component (e.g., 'atm', 'ocn')
    index_hist : int
        Index of the historical scenario to load

    Returns
    -------
    tuple
        A tuple containing two xarray Datasets:
        - ds_var_hist : Dataset for historical period
        - ds_var_fut : Dataset for future scenario
    """
    path_intermed_hist, path_intermed_fut = get_var_paths(directory, var)
    filename_identifier = '.'.join(path_intermed_hist[index_hist].rsplit('.', 5)[1:4])
    index_fut = find_identifier_with_index(path_intermed_hist, filename_identifier)[0][1]
    hist_file_paths = get_hist_file_paths(var, directory, path_intermed_hist, index_hist)
    fut_file_paths = get_fut_file_paths(var, directory, path_intermed_fut, index_fut)
    ds_var_fut = file_path_to_var_ds(fut_file_paths)
    ds_var_hist = file_path_to_var_ds(hist_file_paths)
    return ds_var_hist, ds_var_fut

def find_identifier_with_index(prefixes, identifier):
    """
    Find prefixes that contain a specific identifier and their indices.

    Parameters
    ----------
    prefixes : list
        A list of prefixes to search through
    identifier : str
        The identifier to search for in the prefixes

    Returns
    -------
    list
        A list of tuples containing (matching_prefix, index) pairs
    """
    matching_prefixes_with_indices = []

    for index, prefix in enumerate(prefixes):
        if identifier in prefix:
            matching_prefixes_with_indices.append((prefix, index))

    return matching_prefixes_with_indices


def get_hist_file_paths(var, directory, path_intermed_hist, index):
    """
    Construct file paths for historical simulations.

    Parameters
    ----------
    var : str
        Variable name
    directory : str
        Base directory path
    path_intermed_hist : list
        List of historical scenario prefixes
    index : int
        Index of the scenario to use

    Returns
    -------
    list
        Complete file paths for all historical time chunks:
        - 1850-1859 through 2000-2009
        - 2010-2014
        
    Notes
    -----
    Follows CESM LENS2 historical file naming convention:
    prefix.var.YYYY01-YYYY12.nc
    """
    attrib_title = path_intermed_hist[index]
    file_paths = []
    for start_year in range(1850, 2010, 10):
        end_year = start_year + 9
        file_path = f'{directory}{attrib_title}.{var}.{start_year}01-{end_year}12.nc'
        file_paths.append(file_path)
    last_file_path = f'{directory}{attrib_title}.{var}.201001-201412.nc'
    file_paths.append(last_file_path)
    return file_paths

def get_fut_file_paths(var, directory, path_intermed_fut, index):
    """
    Construct file paths for future simulations.

    Parameters
    ----------
    var : str
        Variable name
    directory : str
        Base directory path
    path_intermed_fut : list
        List of future scenario prefixes
    index : int
        Index of the scenario to use

    Returns
    -------
    list
        Complete file paths for all future time chunks:
        - 2015-2024
        - 2095-2100

    Notes
    -----
    Follows CESM LENS2 SSP370 scenario file naming convention:
    prefix.var.YYYY01-YYYY12.nc
    """
    attrib_title = path_intermed_fut[index]
    file_paths = []
    for start_year in range(2015, 2095, 10):
        end_year = start_year + 9
        file_path = f'{directory}{attrib_title}.{var}.{start_year}01-{end_year}12.nc'
        file_paths.append(file_path)
    last_file_path = f'{directory}{attrib_title}.{var}.209501-210012.nc'
    file_paths.append(last_file_path)
    return file_paths

def file_path_to_var_ds(file_paths):
    """
    Load multiple NetCDF files into a single xarray Dataset.

    Parameters
    ----------
    file_paths : list
        List of file paths to load

    Returns
    -------
    xarray.Dataset
        Dataset concatenated along time dimension
    """
    var_ds = xr.open_mfdataset(file_paths, 
                              concat_dim='time', 
                              combine='nested', 
                              parallel=True)
    return var_ds
