"""
Created on Wed Oct 26 09:09:58 2022

prepare input files and mask areas that are not crops

@author: america
@refactor: woerkom
"""

# %%
import geopandas as gpd
import numpy as np
import xarray as xr
from imod.prepare import rasterize

import utils


# %% functions
def make_mask(file, path_DPZW_regio):
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

# %%
# paths
if "snakemake" not in globals():
    snakefile = r"C:\git_repos\effectmodule_landbouw\snakes\wf_landbouw_test_new"
    snakemake = utils.read_snakemake_rule(snakefile, name="prepare_median_input")

input_fns = snakemake.input
params = snakemake.params
output_fns = snakemake.output

# %% ############################
## mediane opbrengst ref ##
############################
pot_data = xr.open_dataarray(input_fns.pot_data_fn)
act_data = xr.open_dataarray(input_fns.act_data_fn)
median_yield = act_data.median(dim="time")

derv = pot_data - act_data
median_derv = derv.median(dim="time")

# mask voor landbouwgrond en dpzw regios
landgeb = xr.open_dataarray(input_fns.lgn_fn).isel(band=0)
mask_landg = xr.where(landgeb.isin(params.lgn_landbouwgewassen), 1, np.nan)
mask_wad = make_mask(mask_landg, input_fns.path_DPZW_regio)
mask = mask_landg * mask_wad

# mask data
median_yield_masked = median_yield * mask
median_derv_masked = median_derv * mask

#save data
median_yield_masked.rio.to_raster(output_fns.act_med_fn, nodata=-9999)
median_derv_masked.rio.to_raster(output_fns.derv_med_fn, nodata=-9999)

# %%
