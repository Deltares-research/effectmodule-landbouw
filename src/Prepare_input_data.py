"""
Created on Wed Oct 26 09:09:58 2022

prepare input files and mask areas that are not crops

@author: america
@refactor: woerkom
@refactor 2026: reusen
"""

# %%
from pathlib import Path

import xarray as xr

from src import utils
from src.processing.io import write_outputs
from src.processing.spatial.masks import create_crop_mask
from src.processing.yields import calculate_median_yields


def main(snakemake) -> None:
    """
    Prepare median yield input rasters for the agricultural workflow.

    Loads potential and actual yield data, calculates median actual yield
    and yield loss, masks the results to agricultural land within the
    specified regions, and writes the resulting rasters to disk.

    Parameters
    ----------
    snakemake
        Snakemake object containing the input files, output files,
        and parameters required by the workflow.

    Returns
    -------
    None
    """
    pot_data = xr.open_dataarray(snakemake.input.pot_data_fn)

    act_data = xr.open_dataarray(snakemake.input.act_data_fn)

    median_actual, median_derv = calculate_median_yields(
        pot_data,
        act_data,
    )

    lgn = xr.open_dataarray(snakemake.input.lgn_fn).isel(band=0)

    mask = create_crop_mask(
        lgn,
        snakemake.params.lgn_landbouwgewassen,
        snakemake.input.path_DPZW_regio,
    )

    write_outputs(
        median_actual * mask,
        median_derv * mask,
        snakemake.output.act_med_fn,
        snakemake.output.derv_med_fn,
    )


if "snakemake" not in globals():
    snakefile = Path(__file__).parent.parent / "workflows" / "wf_landbouw_test.smk"
    snakemake = utils.read_snakemake_rule(snakefile, name="prepare_median_input")

main(snakemake)
