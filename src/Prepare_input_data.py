# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:09:58 2022

prepare input files and mask areas that are not crops

@author: america
"""

# %%

import numpy as np
from pathlib import Path
import xarray as xr
import imod
import geopandas as gpd

import utils
# %% functions
def make_mask(file, path_DPZW_regio):
    DPZW_regio = gpd.read_file(path_DPZW_regio)
    DPZW_regio["ID"] = DPZW_regio.index
    DPZW_regio_grid = imod.prepare.rasterize(DPZW_regio, like=file, column="ID")
    # waddeneiland = int(DPZW_regio.loc[DPZW_regio['deelregio']=='13 - Noord Waddeneilanden'].ID)
    # file_masked = xr.where(DPZW_regio_grid==waddeneiland, np.nan, file)
    # DPZW_regio_grid_nowad = xr.where(DPZW_regio_grid==waddeneiland, np.nan, DPZW_regio_grid)
    file_masked = xr.where(DPZW_regio_grid >= 0, 1, np.nan)
    return file_masked


def grid_vergroten(da, voorbeeldextent):
    """Na het inladen van .p00x grids met imod-python is het extent van sommige grids te klein,
    omdat deze variabele niet bestaat in een deel van de pks gebieden. Hiervoor wordt
    gecorrigeerd"""
    ## Replace grid to nan
    voorbeeldextent_nan = xr.full_like(voorbeeldextent, np.nan)
    
    ## Vergroot het extent van de data-array
    da_extent_nan = voorbeeldextent_nan.combine_first(da)
    return da_extent_nan


# %%
# paths
if "snakemake" not in globals():
    snakefile = r"P:\11211541-003-kentalleneffecten\Landbouw\011a Effectmodule Landbouw\src\workflow\wf_landbouwv8_debug"
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

# mask enzo
landgeb = imod.rasterio.open(input_fns.lgn_fn)
landbouwgewassen = [1, 2, 3, 4, 5, 6, 7, 9, 10, 21]
mask_landg = xr.where(landgeb.isin(landbouwgewassen), 1, np.nan)
mask_wad = make_mask(mask_landg, input_fns.path_DPZW_regio)
mask = mask_landg * mask_wad

# mask data and fix grid shape
median_yield_masked = median_yield * mask
median_yield_masked = grid_vergroten(median_yield_masked, mask)

median_derv_masked = median_derv * mask
median_derv_masked = grid_vergroten(median_derv_masked, mask)

print(output_fns.act_med_fn)
imod.rasterio.save(output_fns.act_med_fn, median_yield_masked, nodata=-9999)
imod.rasterio.save(output_fns.derv_med_fn, median_derv_masked, nodata=-9999)

#imod.rasterio.save(mask_wad_fn, mask_wad, nodata=-9999)

# %%
#test
# original_masked_median = r"N:\Projects\11210000\11210362\B. Measurements and calculations\011a Effectmodule Landbouw\data\old_scripts\REF2017BP18\masked_median_ActYld.asc"
# original_masked_median = imod.rasterio.open(original_masked_median)

# original_masked_median_filled = original_masked_median.fillna(-9999)
# median_yield_masked_filled = median_yield_masked.fillna(-9999)

# assert (original_masked_median_filled == median_yield_masked_filled).all().compute()
# %%
