import xarray as xr


def calculate_variable_irrigation_costs(
    arbeidskosten_sce: xr.DataArray,
    energiekosten_sce: xr.DataArray,
    perc_prijspeil_beregening: float,
) -> xr.DataArray:
    """
    Calculate variable irrigation costs above the median cost level.

    Labour and energy costs are combined, doubled, and adjusted for
    the irrigation price level. Only irrigation costs above the median
    over all years are retained.

    Parameters
    ----------
    arbeidskosten_sce : xr.DataArray
        Scenario labour costs.
    energiekosten_sce : xr.DataArray
        Scenario energy costs.
    perc_prijspeil_beregening : float
        Percentage adjustment applied to the irrigation cost price level.

    Returns
    -------
    xr.DataArray
        Variable irrigation costs above the median cost level.
    """
    variable_irrigation_costs = (
        (1 + (perc_prijspeil_beregening / 100))
        * 2
        * (arbeidskosten_sce + energiekosten_sce)
    )

    variable_irrigation_costs_median = variable_irrigation_costs.median(dim="Year")

    variable_irrigation_costs_median_broadcasted = xr.concat(
        [
            variable_irrigation_costs_median.expand_dims(Year=[year])
            for year in variable_irrigation_costs.coords["Year"].values
        ],
        dim="Year",
    )

    variable_irrigation_costs_def = (
        variable_irrigation_costs - variable_irrigation_costs_median_broadcasted
    ).where(
        variable_irrigation_costs >= variable_irrigation_costs_median_broadcasted,
        0,
    )

    return variable_irrigation_costs_def


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
