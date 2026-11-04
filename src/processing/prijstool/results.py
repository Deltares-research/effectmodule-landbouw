import pandas as pd
import xarray as xr


def create_results_dataframe(
    standard_opbrengst_kg_sce: xr.DataArray,
    droogtederving_kg_sce: xr.DataArray,
    change_producer_surplus: xr.DataArray,
    change_consumer_surplus: xr.DataArray,
    change_total_welfare: xr.DataArray,
    variable_irrigation_costs_def: xr.DataArray,
    areaal_sce: xr.DataArray,
    base_price: xr.DataArray,
    new_price: xr.DataArray,
    area_name_num: pd.Series,
) -> pd.DataFrame:
    """
    Create the final Prijstool results DataFrame.

    Combines production, production loss, welfare effects, irrigation
    costs, area, and price data into a single DataFrame. Region numbers
    are replaced by region names.

    Parameters
    ----------
    standard_opbrengst_kg_sce : xr.DataArray
        Standard scenario production.
    droogtederving_kg_sce : xr.DataArray
        Scenario production loss due to drought.
    change_producer_surplus : xr.DataArray
        Change in producer surplus.
    change_consumer_surplus : xr.DataArray
        Change in consumer surplus.
    change_total_welfare : xr.DataArray
        Change in total welfare.
    variable_irrigation_costs_def : xr.DataArray
        Variable irrigation costs above the median cost level.
    areaal_sce : xr.DataArray
        Scenario agricultural area.
    base_price : xr.DataArray
        Base crop prices.
    new_price : xr.DataArray
        New crop prices after accounting for production changes.
    area_name_num : pd.Series
        Mapping from region numbers to region names.

    Returns
    -------
    pd.DataFrame
        Final Prijstool results, indexed by crop, area, and year.
    """
    merge_data = [
        standard_opbrengst_kg_sce,
        droogtederving_kg_sce,
        change_producer_surplus,
        change_consumer_surplus,
        change_total_welfare,
        variable_irrigation_costs_def,
        areaal_sce,
        base_price.broadcast_like(new_price),
        new_price,
    ]

    merge_columns = [
        "productie",
        "productiederving",
        "veranderingen producentensurplus",
        "verandering consumenten surplus",
        "verandering totale welvaart",
        "variabele beregeningskosten",
        "areaal",
        "prijs",
        "nieuwe prijs",
    ]

    comb1 = xr.merge([x.rename(n) for x, n in zip(merge_data, merge_columns)])

    area_names = area_name_num[comb1.Area.values].values

    comb1 = comb1.assign_coords(Area=area_names)

    comb1_df = comb1.to_dataframe().sort_index()

    return comb1_df


def aggregate_welfare_by_area_and_year(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total welfare change by area and year."""
    return results["verandering totale welvaart"].unstack("Area").groupby("Year").sum()


def aggregate_welfare_by_crop_and_year(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate total welfare change by crop and year."""
    return results["verandering totale welvaart"].unstack("Crop").groupby("Year").sum()


def aggregate_welfare_by_crop_and_area(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate average total welfare change by crop and area."""
    return results["verandering totale welvaart"].unstack("Crop").groupby("Area").mean()
