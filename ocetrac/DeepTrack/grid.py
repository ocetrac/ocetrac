# ============================================================
# DeepTrack/grid.py
# ============================================================
"""
Grid helpers: depth geometry, cell volume, and structuring elements.

Supported grids
---------------
POP_grid : curvilinear, TAREA in cm², z_t in cm  ← implemented
MOM_grid : TODO
NEMO_grid: TODO
isopycnal: TODO
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def compute_dz(z_t: np.ndarray) -> np.ndarray:
    """
    Convert depth midpoints to layer thicknesses via central differencing;
    forward/backward differencing at the boundaries.

    Parameters
    ----------
    z_t : 1-D array of depth midpoints (e.g. cm for POP grids)

    Returns
    -------
    dz : 1-D array, same unit as z_t
    """
    z  = np.asarray(z_t, dtype=float)
    dz = np.zeros_like(z)
    if len(z) == 1:
        return dz
    dz[1:-1] = (z[2:] - z[:-2]) / 2.0
    dz[0]    = z[1]  - z[0]
    dz[-1]   = z[-1] - z[-2]
    return dz


def build_cell_volume(
    TAREA: xr.DataArray,
    z_t:   xr.DataArray,
    n_z:   int = 20,
) -> xr.DataArray:
    """
    Compute a 3-D cell-volume array (z_t, nlat, nlon) for a POP-style grid.

    Expects POP units: TAREA in cm², z_t midpoints in cm.
    Output is in m³.

    Parameters
    ----------
    TAREA : DataArray (nlat, nlon) — grid-cell area in cm²
    z_t   : DataArray (z_t,)       — depth midpoints in cm
    n_z   : int                    — number of depth levels to use

    Returns
    -------
    cell_volume : DataArray (z_t, nlat, nlon) in m³
    """
    TAREA_m2 = TAREA * 1e-4
    dz_m     = compute_dz(z_t[:n_z].values) / 100.0
    dz_da    = xr.DataArray(dz_m, dims=["z_t"], coords={"z_t": z_t[:n_z]})
    return (TAREA_m2 * dz_da).transpose("z_t", "nlat", "nlon")


def make_anisotropic_struct(
    connect_xy: bool = True,
    connect_z:  bool = True,
) -> np.ndarray:
    """
    Build a (3, 3, 3) boolean structuring element for 3-D connected-component
    labelling. Axis order: (z, y, x).

    - Horizontal (y, x) plane: full 8-connectivity (queen's move).
    - Vertical z: face-only (no diagonal steps in z).
    - Either direction can be disabled independently.

    Parameters
    ----------
    connect_xy : bool — include horizontal neighbours
    connect_z  : bool — include vertical neighbours

    Returns
    -------
    struct : bool ndarray (3, 3, 3)
    """
    struct = np.zeros((3, 3, 3), dtype=bool)
    struct[1, 1, 1] = True

    if connect_xy:
        struct[1, 1, 0] = struct[1, 1, 2] = True
        struct[1, 0, 1] = struct[1, 2, 1] = True
        struct[1, 0, 0] = struct[1, 0, 2] = True
        struct[1, 2, 0] = struct[1, 2, 2] = True

    if connect_z:
        struct[0, 1, 1] = struct[2, 1, 1] = True

    return struct
