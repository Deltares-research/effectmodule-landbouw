import geopandas as gpd
import imod
import numpy as np
import xarray as xr


def create_region_grid(
    region_shapefile: str,
    like: xr.DataArray,
) -> xr.DataArray:
    """
    Rasterize region polygons to the grid of a reference raster.

    Parameters
    ----------
    region_shapefile : str
        Path to the region shapefile.
    like : xr.DataArray
        Raster whose grid, extent and resolution will be used.

    Returns
    -------
    xr.DataArray
        Rasterized region grid.
    """
    regions = gpd.read_file(region_shapefile)

    regions["Nummer"] = regions["Nummer"].astype(float)

    region_grid = imod.prepare.rasterize(
        geodataframe=regions,
        like=like,
        column="Nummer",
        fill=np.nan,
    )

    return region_grid
