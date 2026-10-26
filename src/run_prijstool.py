# %%
from pathlib import Path

import pandas as pd

import utils
from processing.prijstool.model import Prijstool


def main(snakemake) -> None:
    """
    Run the Prijstool economic impact calculations.

    Loads the reference and scenario crop data, elasticity data,
    and crop prices, initialises the Prijstool model, runs the
    calculations, and writes the model outputs.

    Parameters
    ----------
    snakemake
        Snakemake object containing the input files, output files,
        and parameters required by the workflow.
    """

    # load crop data
    cropdata_split_ref = pd.read_csv(snakemake.input.cropdata_split_ref_fn, index_col=0)
    cropdata_split_sce = pd.read_csv(snakemake.input.cropdata_split_sce_fn, index_col=0)

    prijstool_elasticity = pd.read_csv(snakemake.input.elasticity_fn, index_col=0)
    prijstool_elasticity = prijstool_elasticity.drop("gewashoofd").astype(float)

    crop_prices = pd.read_csv(snakemake.input.crop_prices_fn, index_col=0)

    prijstool = Prijstool(
        prijstool_elasticity,
        crop_prices,
        cropdata_split_sce.copy(),
        cropdata_split_ref.copy(),
        snakemake.params,
    )

    prijstool.run()
    prijstool.save_output(snakemake.output)
    # %%


if "snakemake" not in globals():
    snakefile = Path(__file__).parent.parent / "workflows" / "wf_landbouw_test.smk"
    snakemake = utils.read_snakemake_rule(snakefile, name="run_prijstool")

main(snakemake)
