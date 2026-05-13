"""
temporal_measures.py
"""

import datetime

import cftime
import numpy as np
import pandas as pd

from cftime import num2date


def get_initial_detection_time(one_obj):
    """
    Gets the initial detection time from an xarray object.

    Parameters
    ----------
    one_obj : xarray.DataArray or xarray.Dataset
        The xarray object containing a 'time' coordinate.

    Returns
    -------
    t0
        The initial detection time the same type as one_obj.time

    Raises
    ------
    ValueError
        If the input dataset has an empty time coordinate.
    """
    if one_obj.time.size == 0:
        raise ValueError("Input dataset has empty time coordinate")

    t0 = one_obj.time[0].item()

    return t0


def get_duration(one_obj):
    """
    Gets the duration of the time coordinate in an xarray object.

    Parameters
    ----------
    one_obj : xarray.DataArray or xarray.Dataset
        The xarray object containing a 'time' coordinate.

    Returns
    -------
    int
        The number of time steps in the time coordinate.
    """
    return one_obj.time.shape[0]
