import xarray as xr


def calibrate_constant_elasticity_demand(
    standard_production_ref: xr.DataArray,
    standard_production_ref_world: xr.DataArray,
    base_price: xr.DataArray,
    elasticity: xr.Dataset,
    local_market_price_effect: xr.DataArray,
    world_market_price_effect: xr.DataArray,
) -> xr.DataArray:
    """
    Calibrate a constant-elasticity demand function.

    The demand function has the form y = a * p**b, where ``a`` is
    the demand scale factor and ``b`` is the price elasticity.

    Parameters
    ----------
    standard_production_ref : xr.DataArray
        Standard reference production by crop and area.
    standard_production_ref_world : xr.DataArray
        Standard reference production by crop for the world market.
    base_price : xr.DataArray
        Base price by crop and area.
    elasticity : xr.Dataset
        Elasticity data containing the ``elasticity`` variable.
    local_market_price_effect : xr.DataArray
        Boolean mask for crops with a local-market price effect.
    world_market_price_effect : xr.DataArray
        Boolean mask for crops with a world-market price effect.

    Returns
    -------
    xr.DataArray
        Demand scale factor for the constant-elasticity demand function.
    """
    demand_scale_factor = xr.where(
        local_market_price_effect,
        standard_production_ref / (base_price ** elasticity["elasticity"]),
        0,
    )

    demand_scale_factor = xr.where(
        world_market_price_effect,
        standard_production_ref_world / (base_price ** elasticity["elasticity"]),
        demand_scale_factor,
    )

    demand_scale_factor = xr.where(
        demand_scale_factor == 0,
        1,
        demand_scale_factor,
    )

    return demand_scale_factor


def calibrate_linear_demand(
    standard_production_ref: xr.DataArray,
    standard_production_ref_world: xr.DataArray,
    base_price: xr.DataArray,
    elasticity: xr.Dataset,
    local_market_price_effect: xr.DataArray,
    world_market_price_effect: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calibrate a linear demand function.

    The demand function has the form
    ``y = demand_slope * price + demand_intercept``.

    Parameters
    ----------
    standard_production_ref : xr.DataArray
        Standard reference production by crop and area.
    standard_production_ref_world : xr.DataArray
        Standard reference production by crop for the world market.
    base_price : xr.DataArray
        Base price by crop and area.
    elasticity : xr.Dataset
        Elasticity data containing the ``elasticity`` variable.
    local_market_price_effect : xr.DataArray
        Boolean mask for crops with a local-market price effect.
    world_market_price_effect : xr.DataArray
        Boolean mask for crops with a world-market price effect.

    Returns
    -------
    tuple of xr.DataArray
        Demand slope and demand intercept.
    """
    demand_slope = xr.where(
        local_market_price_effect,
        elasticity["elasticity"] * standard_production_ref / base_price,
        1,
    )

    demand_slope = xr.where(
        world_market_price_effect,
        elasticity["elasticity"] * standard_production_ref_world / base_price,
        demand_slope,
    )

    demand_slope = xr.where(
        demand_slope == 0,
        1,
        demand_slope,
    )

    demand_intercept = xr.where(
        local_market_price_effect,
        standard_production_ref - demand_slope * base_price,
        0,
    )

    demand_intercept = xr.where(
        world_market_price_effect,
        standard_production_ref_world - demand_slope * base_price,
        demand_intercept,
    )

    return demand_slope, demand_intercept
