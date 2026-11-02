import xarray as xr


def calculate_welfare_changes(
    variable_irrigation_costs_def: xr.DataArray,
    new_yield_local: xr.DataArray,
    new_price: xr.DataArray,
    standard_opbrengst_kg_sce: xr.DataArray,
    base_price: xr.DataArray,
    droogtederving_kg_sce: xr.DataArray,
    change_consumer_surplus: xr.DataArray,
    p_elas: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate changes in producer surplus and total welfare.

    Parameters
    ----------
    variable_irrigation_costs_def : xr.DataArray
        Variable irrigation costs above the median cost level.
    new_yield_local : xr.DataArray
        Local production after yield losses.
    new_price : xr.DataArray
        New crop prices after accounting for production changes.
    standard_opbrengst_kg_sce : xr.DataArray
        Standard scenario production.
    base_price : xr.DataArray
        Base crop prices.
    droogtederving_kg_sce : xr.DataArray
        Scenario production loss due to drought.
    change_consumer_surplus : xr.DataArray
        Change in consumer surplus.
    p_elas : xr.Dataset
        Elasticity data containing the price-effect classification.

    Returns
    -------
    tuple of xr.DataArray
        Change in producer surplus and change in total welfare.
    """
    change_producer_surplus = -variable_irrigation_costs_def + xr.where(
        p_elas["prijseffect"] > 0,
        ((new_yield_local * new_price) - (standard_opbrengst_kg_sce * base_price)),
        -(droogtederving_kg_sce * base_price),
    )

    change_total_welfare = change_producer_surplus + change_consumer_surplus

    return (
        change_producer_surplus,
        change_total_welfare,
    )
