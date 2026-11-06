# %%
# Load necessary libraries
from pathlib import Path

import pandas as pd

import utils
from processing.spatial.regions import create_region_mapping


def aggregate_drought_risk_by_main_region(
    region_data: pd.DataFrame,
    region_mapping: pd.Series,
) -> pd.DataFrame:
    """
    Aggregate drought risk from subregions to main regions.

    Parameters
    ----------
    region_data : pd.DataFrame
        Drought-risk data indexed by subregion, including a
        'Scenario' column.
    region_mapping : pd.Series
        Main-region names indexed by subregion name.

    Returns
    -------
    pd.DataFrame
        Drought risk per main region and scenario, including
        the total across all main regions, expressed in million euros.
    """
    region_data["Hoofdregio"] = region_mapping[region_data.index].values

    main_region_risk = region_data.groupby(["Hoofdregio", "Scenario"]).sum()

    total = main_region_risk.groupby("Scenario").sum()
    total["Hoofdregio"] = "Totaal"

    total = total.reset_index().set_index(["Hoofdregio", "Scenario"])

    drought_risk = pd.concat([main_region_risk, total])

    drought_risk /= 1e6

    drought_risk.columns = [
        "Droogterisico (mln euro / jaar)",
        "Droogterisico_T20 (mln euro)",
    ]

    return drought_risk


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

    region_data = pd.read_excel(
        snakemake.input.droogterisico_fn,
        index_col=[0, 1],
    )
    region_data = region_data.groupby("Area").sum()
    region_data["Scenario"] = snakemake.params.scenario

    region_data["Hoofdregio"] = region_mapping[region_data.index].values

    drought_risk = aggregate_drought_risk_by_main_region(
        region_data,
        region_mapping,
    )

    drought_risk.to_excel(snakemake.output.droogterisico_hoofdregios_fn)


if "snakemake" not in globals():
    snakefile = Path(__file__).parent.parent / "workflows" / "wf_landbouw_test.smk"

    snakemake = utils.read_snakemake_rule(
        snakefile,
        name="risico_hoofdregio",
    )

main(snakemake)
