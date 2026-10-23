import imod
import xarray as xr


def load_input_data(input_fns):
    """
    Load raster input data for Prijstool preprocessing.

    Parameters
    ----------
    input_fns
        Snakemake input object containing the input file paths.

    Returns
    -------
    dict
        Dictionary containing the loaded input datasets.
    """
    data = {
        "Yldactdry_med_combi": imod.rasterio.open(input_fns.Yldactdry_med_combi_fn),
        "Ylderv_med_combi": imod.rasterio.open(input_fns.Dervactdry_med_combi_fn),
        "potASC_combi": xr.open_dataarray(input_fns.potASC_combi_fn),
        "act_dry_ASC_combi": xr.open_dataarray(input_fns.act_dry_ASC_combi_fn),
        "landgebruik_ref": imod.rasterio.open(input_fns.landgebruik_ref_fn),
        "areaalasc_ref": imod.rasterio.open(input_fns.areaalasc_ref_fn),
        "landgebruik_sce": xr.open_dataarray(input_fns.landgebruik_fn).rename(
            "landgebruik"
        ),
        "areaalasc_sce": imod.rasterio.open(input_fns.areaalasc_fn).rename("areaal"),
        "arbdASC_sce": xr.open_dataarray(input_fns.arbdASC_fn),
        "enrgASC_sce": xr.open_dataarray(input_fns.enrgASC_fn),
    }

    return data


def write_outputs(median_actual_masked, median_derv_masked, act_med_fn, derv_med_fn):
    median_actual_masked.rio.to_raster(act_med_fn, nodata=-9999)
    median_derv_masked.rio.to_raster(derv_med_fn, nodata=-9999)
