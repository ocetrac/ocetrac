import cftime
from cftime import num2date
import numpy as np
import pandas as pd

def get_initial_detection_time(one_obj, units=None, calendar="noleap"):
    if one_obj.time.size == 0:
        raise ValueError("Input dataset has empty time coordinate")

    t0 = one_obj.time[0].item()

    if isinstance(t0, cftime.DatetimeNoLeap):
        year, month, day = t0.year, t0.month, t0.day
    elif isinstance(t0, np.datetime64):
        dt = pd.to_datetime(str(t0))
        year, month, day = dt.year, dt.month, dt.day
    elif isinstance(t0, (pd.Timestamp, datetime.datetime)):
        year, month, day = t0.year, t0.month, t0.day
    elif isinstance(t0, (int, float)):

        if units is None:
            raise ValueError("Provide 'units' string for numeric time values")
        t0_date = num2date(float(t0), units=units, calendar=calendar)
        year, month, day = t0_date.year, t0_date.month, t0_date.day

    else:
        raise TypeError(f"Unsupported time type: {type(t0)}")

    return cftime.DatetimeNoLeap(year, month, day)


def get_duration(one_obj):
    return one_obj.time.shape[0]