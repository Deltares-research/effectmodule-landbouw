"""
Created on Mon Apr 23 09:48:43 2018
Prijstool
converter: https://bit.ly/2r2JILl
Format the pandas file as GAMS scv inputs
- Possibility to split
- Possibility to read idf file, note: library raster_func from Wiebe Borren required!
@author: hermawan

The workflow:
Raster outputs
    ↓
Aggregate per crop + region
    ↓
Calculate damages
    ↓
Add costs
    ↓
Split crops if needed
    ↓
Write Prijstool/GAMS input CSV

Earlier comments for the functions:
using idf as input files
MvdV 191030 added reading of areaal.asc and multiplication with areaal / 10000
MvdV 191128 added switch perha to muliply with ha per cell or not
JP 200619 replacing raster-func with imod-python to open grids if AGRICOM output is in idfs.
the asc file is loaded as 2 dimensional array

Version histrory
v8   259219 Teun van Woerkom translated script to xarray dataformat and numpy for name-dependent calculation instead of index-dependent
v7   250128 Marnix van der Vat modified option 'optdervAct_Actmed' to calculate economic impact relative to the minimum of potental and median production
1.04 191030 Marnix van der Vat reads areaal.asc and multiplies input values with areaal/10000 to convert input from unit/ha to unit instead of multiplying with 6.25
1.05 191119 Marnix van der Vat doubling of variable irrigation costs
1.06 1911281128 Marnix van der Vat correction of multiplication of energy labour costs with the area in ha per grid cel
1.061 191204 Marnix van der Vat remoeplaced Pandas from_csv with read_csv to comply with newer Pandas versions
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import utils
from src.processing.io import load_input_data
from src.processing.prijstool.cropdata import create_cropdata, finalise_cropdata
from src.processing.prijstool.splitting import apply_crop_splitting
from src.processing.prijstool.yield_loss import calculate_yield_loss
from src.processing.spatial.regions import create_region_grid
from src.processing.validation import validate_continuous_years


def main(snakemake) -> None:
    """
    Create input data for the Prijstool workflow.

    Loads the required raster data, validates the simulation years,
    aggregates data by crop and region, calculates yield losses,
    and writes the regular and crop-split Prijstool input files.

    Parameters
    ----------
    snakemake
        Snakemake object containing the input files, output files,
        and parameters for the workflow.
    """

    # Prepare output directories
    Path(snakemake.output.cropdata_fn).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(snakemake.output.cropdata_split_fn).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load input data
    data = load_input_data(snakemake.input)

    potASC_combi = data["potASC_combi"]
    act_dry_ASC_combi = data["act_dry_ASC_combi"]
    areaalasc_sce = data["areaalasc_sce"]

    # Create region grid
    regios_ar = create_region_grid(
        snakemake.input.risico_subregio_fn,
        areaalasc_sce,
    )

    # Validate years
    years = np.unique(potASC_combi.time.dt.year)

    validate_continuous_years(years)

    # Create crop database
    cropdata = create_cropdata(
        potASC_combi=potASC_combi,
        act_dry_ASC_combi=act_dry_ASC_combi,
        Ylderv_med_combi=data["Ylderv_med_combi"],
        arbdASC_sce=data["arbdASC_sce"],
        enrgASC_sce=data["enrgASC_sce"],
        landgebruik_sce=data["landgebruik_sce"],
        landgebruik_ref=data["landgebruik_ref"],
        regios_ar=regios_ar,
        areaalasc_sce=areaalasc_sce,
        areaalasc_ref=data["areaalasc_ref"],
        years=years,
    )

    # Calculate yield loss
    cropdata = calculate_yield_loss(
        cropdata,
        method=snakemake.params.calc,
    )

    # Finalise crop database
    cropdata = finalise_cropdata(
        cropdata,
        scenario=snakemake.params.scenario,
    )

    cropdata.to_csv(
        snakemake.output.cropdata_fn,
    )

    # Split crops
    sub_crops = pd.read_csv(
        snakemake.input.sub_crops_fn,
        index_col=0,
    )

    cropdata_split = apply_crop_splitting(
        cropdata,
        sub_crops,
    )

    cropdata_split.to_csv(
        snakemake.output.cropdata_split_fn,
    )


if "snakemake" not in globals():
    snakefile = Path(__file__).parent.parent / "workflows" / "wf_landbouw_test.smk"
    snakemake = utils.read_snakemake_rule(snakefile, name="run_prijstool_preprocessing")

main(snakemake)
