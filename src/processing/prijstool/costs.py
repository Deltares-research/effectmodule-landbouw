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
