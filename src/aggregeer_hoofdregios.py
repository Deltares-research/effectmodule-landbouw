# %%
# Load necessary libraries
from pathlib import Path

import pandas as pd

import utils
from processing.spatial.regions import create_region_mapping


def main(snakemake) -> None:
    """
    Aggregate drought risk from subregions to main regions.

    Loads drought-risk results and region geometries, assigns each
    subregion to a main region, aggregates drought risk per main region,
    adds the total across all regions, and writes the results to Excel.

    Parameters
    ----------
    snakemake
        Snakemake object containing the input and output files required
        by the workflow.
    """

    region_mapping = create_region_mapping(
        snakemake.input.subregio_fn,
        snakemake.input.hoofdregio_fn,
    )

    regio_df = pd.read_excel(
        snakemake.input.droogterisico_fn,
        index_col=[0, 1],
    )
    regio_df = regio_df.groupby("Area").sum()
    regio_df["Scenario"] = "REF2017BP18"

    regio_df["Hoofdregio"] = region_mapping[regio_df.index].values

    hoofdregio_droogterisico = regio_df.groupby(["Hoofdregio", "Scenario"]).sum()

    totaal = hoofdregio_droogterisico.groupby("Scenario").sum()
    totaal["Hoofdregio"] = "Totaal"
    totaal = totaal.reset_index().set_index(["Hoofdregio", "Scenario"])
    droogte_risico_df = pd.concat([hoofdregio_droogterisico, totaal])
    droogte_risico_df /= 1e6

    droogte_risico_df.columns = [
        "Droogterisico (mln euro / jaar)",
        "Droogterisico_T20 (mln euro)",
    ]
    droogte_risico_df.to_excel(snakemake.output.droogterisico_hoofdregios_fn)


if "snakemake" not in globals():
    snakefile = Path(__file__).parent.parent / "workflows" / "wf_landbouw_test.smk"

    snakemake = utils.read_snakemake_rule(
        snakefile,
        name="risico_hoofdregio",
    )

main(snakemake)
