# %% functions
import geopandas as gpd
import numpy as np
import xarray as xr
from imod.prepare import rasterize


def create_dpzw_mask(file, path_DPZW_regio):
    """
    Mask a given file based on the DPZW regio shapefile.

    Parameters
    ----------
    file : xarray.DataArray
        The data array to be masked.
    path_DPZW_regio : str
        Path to the DPZW regio shapefile.

    Returns
    -------
    xarray.DataArray
        The masked data array.
    """
    DPZW_regio = gpd.read_file(path_DPZW_regio)
    DPZW_regio["ID"] = DPZW_regio.index
    DPZW_regio_grid = rasterize(DPZW_regio, like=file, column="ID")
    file_masked = xr.where(DPZW_regio_grid >= 0, 1, np.nan)
    return file_masked


def create_crop_mask(
    lgn_raster,
    crop_codes,
    dpzw_path,
):
    crop_mask = xr.where(
        lgn_raster.isin(crop_codes),
        1,
        np.nan,
    )

    dpzw_mask = create_dpzw_mask(
        crop_mask,
        dpzw_path,
    )

    return crop_mask * dpzw_mask
