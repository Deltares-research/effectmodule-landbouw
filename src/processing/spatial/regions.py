import geopandas as gpd
import imod
import numpy as np
import pandas as pd
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


def create_region_mapping(
    subregion_path,
    main_region_path,
) -> pd.Series:
    """
    Map subregions to main regions using a spatial join.

    Parameters
    ----------
    subregion_path
        Path to the subregion geometry file.
    main_region_path
        Path to the main-region geometry file.

    Returns
    -------
    pd.Series
        Main-region name indexed by subregion name.
    """
    subregions = gpd.read_file(subregion_path)
    subregions["geometry"] = subregions.representative_point()

    main_regions = gpd.read_file(main_region_path)

    joined_regions = gpd.sjoin(
        subregions,
        main_regions,
        how="left",
        predicate="within",
    )

    region_mapping = pd.Series(
        joined_regions["regionaam"].values,
        index=joined_regions["Naam"].values,
    )

    return region_mapping
