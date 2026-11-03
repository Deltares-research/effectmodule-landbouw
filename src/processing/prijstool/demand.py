import numpy as np
import pandas as pd
import xarray as xr


def prepare_elasticity(
    elasticity: pd.DataFrame,
    crop_names: list,
) -> xr.Dataset:
    """
    Prepare elasticity data for the Prijstool calculations.

    Parameters
    ----------
    elasticity : pd.DataFrame
        Prijstool elasticity data.
    crop_names : list
        Crop names included in the calculation.

    Returns
    -------
    xr.Dataset
        Elasticity data indexed by crop.
    """

    # Load data for 'single' crops : Grasland, mais, bieten, granen, boomteelt, fruit
    elasticity_single_crops = elasticity[crop_names]

    elasticity_single_crops.columns.name = "Crop"

    elasticity_xr = elasticity_single_crops.T.sort_index().to_xarray()

    elasticity_xr[
        [
            "NL aandeel wereldaanbod",
            "NL aandeel wereldvraag",
        ]
    ] = xr.where(
        elasticity_xr[
            [
                "NL aandeel wereldaanbod",
                "NL aandeel wereldvraag",
            ]
        ]
        <= 0,
        1,
        elasticity_xr[
            [
                "NL aandeel wereldaanbod",
                "NL aandeel wereldvraag",
            ]
        ],
    )

    return elasticity_xr


def get_price_effect_masks(elasticity: xr.Dataset):
    """
    Create masks for the different price-effect categories.

    Parameters
    ----------
    elasticity : xr.Dataset
        Elasticity data containing the 'prijseffect' variable.

    Returns
    -------
    tuple of xr.DataArray
        Masks for fixed-price, local-market, and world-market
        price effects.
    """
    fixed_price_effect = elasticity["prijseffect"] == 0
    local_market_price_effect = elasticity["prijseffect"] == 1
    world_market_price_effect = elasticity["prijseffect"] == 2

    return (
        fixed_price_effect,
        local_market_price_effect,
        world_market_price_effect,
    )


def prepare_base_prices(
    crop_prices: pd.DataFrame,
    crop_names: list[str],
) -> xr.DataArray:
    """
    Prepare base crop prices for the Prijstool calculations.

    Parameters
    ----------
    crop_prices : pd.DataFrame
        Crop prices per region, including a 'regio_nummer' column.
    crop_names : list[str]
        Names of the crops included in the calculation.

    Returns
    -------
    xr.DataArray
        Base prices indexed by crop and area.
    """
    base_price = crop_prices.set_index("regio_nummer")
    base_price = base_price[crop_names]
    base_price = base_price.stack(0)
    base_price.index.names = ["Area", "Crop"]
    base_price = base_price.swaplevel(0, 1).sort_index()
    base_price = base_price.to_xarray()

    return base_price


def calibrate_constant_elasticity_demand(
    standard_production: xr.DataArray,
    standard_production_world: xr.DataArray,
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
    standard_production : xr.DataArray
        Standard reference production by crop and area.
    standard_production_world : xr.DataArray
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
        standard_production / (base_price ** elasticity["elasticity"]),
        0,
    )

    demand_scale_factor = xr.where(
        world_market_price_effect,
        standard_production_world / (base_price ** elasticity["elasticity"]),
        demand_scale_factor,
    )

    demand_scale_factor = xr.where(
        demand_scale_factor == 0,
        1,
        demand_scale_factor,
    )

    return demand_scale_factor


def calibrate_linear_demand(
    standard_production: xr.DataArray,
    standard_production_world: xr.DataArray,
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
    standard_production : xr.DataArray
        Standard reference production by crop and area.
    standard_production_world : xr.DataArray
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
        elasticity["elasticity"] * standard_production / base_price,
        1,
    )

    demand_slope = xr.where(
        world_market_price_effect,
        elasticity["elasticity"] * standard_production_world / base_price,
        demand_slope,
    )

    demand_slope = xr.where(
        demand_slope == 0,
        1,
        demand_slope,
    )

    demand_intercept = xr.where(
        local_market_price_effect,
        standard_production - demand_slope * base_price,
        0,
    )

    demand_intercept = xr.where(
        world_market_price_effect,
        standard_production_world - demand_slope * base_price,
        demand_intercept,
    )

    return demand_slope, demand_intercept


def calculate_constant_elasticity_price_and_consumer_surplus(
    new_yield_local: xr.DataArray,
    new_yield_world_per_crop_year: xr.DataArray,
    demand_scale_factor: xr.DataArray,
    base_price: xr.DataArray,
    p_elas: xr.Dataset,
    fixed_price_effect: xr.DataArray,
    local_market_price_effect: xr.DataArray,
    world_market_price_effect: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate new prices and consumer surplus using constant elasticity.

    Parameters
    ----------
    new_yield_local : xr.DataArray
        New local yield after production losses.
    new_yield_world_per_crop_year : xr.DataArray
        New world-market yield by crop and year.
    demand_scale_factor : xr.DataArray
        Scale factor of the constant-elasticity demand function.
    base_price : xr.DataArray
        Base price by crop and area.
    p_elas : xr.Dataset
        Elasticity data used in the price calculation.
    fixed_price_effect : xr.DataArray
        Boolean mask for crops with a fixed price effect.
    local_market_price_effect : xr.DataArray
        Boolean mask for crops with a local-market price effect.
    world_market_price_effect : xr.DataArray
        Boolean mask for crops with a world-market price effect.

    Returns
    -------
    tuple of xr.DataArray
        New price and change in consumer surplus.
    """
    new_yield_local = xr.where(
        new_yield_local == 0,
        1,
        new_yield_local,
    )

    new_yield_world_per_crop_year = xr.where(
        new_yield_world_per_crop_year == 0,
        1,
        new_yield_world_per_crop_year,
    )

    new_price = base_price.copy()

    new_price = xr.where(
        local_market_price_effect,
        (new_yield_local / demand_scale_factor) ** (1 / p_elas["elasticity"]),
        new_price,
    )

    new_price = xr.where(
        world_market_price_effect,
        (new_yield_world_per_crop_year / demand_scale_factor)
        ** (1 / p_elas["elasticity"]),
        new_price,
    )

    # Prevent division by zero when elasticity equals -1.
    alt_elasd = xr.where(
        p_elas["elasticity"] == -1,
        -9,
        p_elas["elasticity"],
    )

    change_consumer_surplus = xr.where(
        p_elas["elasticity"] == -1,
        demand_scale_factor * np.log(base_price)
        - demand_scale_factor * np.log(new_price),
        (demand_scale_factor / (alt_elasd + 1) * base_price ** (alt_elasd + 1))
        - (demand_scale_factor / (alt_elasd + 1) * new_price ** (alt_elasd + 1)),
    )

    change_consumer_surplus = xr.where(
        world_market_price_effect,
        p_elas["NL aandeel wereldvraag"] * change_consumer_surplus,
        change_consumer_surplus,
    )

    change_consumer_surplus.loc[fixed_price_effect] = 0

    return new_price, change_consumer_surplus


def calculate_linear_price_and_consumer_surplus(
    new_yield_local: xr.DataArray,
    new_yield_world: xr.DataArray,
    reconstructed_local_yield: xr.DataArray,
    standard_production_sce: xr.DataArray,
    demand_slope: xr.DataArray,
    demand_intercept: xr.DataArray,
    base_price: xr.DataArray,
    local_market_price_effect: xr.DataArray,
    world_market_price_effect: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate new prices and the change in consumer surplus using a
    linear demand function.

    Parameters
    ----------
    new_yield_local : xr.DataArray
        Local production after scenario yield losses.
    new_yield_world : xr.DataArray
        World-market production after scenario yield losses.
    reconstructed_local_yield : xr.DataArray
        Local production reconstructed from world-market production.
    standard_production_sce : xr.DataArray
        Standard scenario production before yield losses.
    demand_slope : xr.DataArray
        Slope of the calibrated linear demand function.
    demand_intercept : xr.DataArray
        Intercept of the calibrated linear demand function.
    base_price : xr.DataArray
        Base price by crop and area.
    local_market_price_effect : xr.DataArray
        Boolean mask for crops with a local-market price effect.
    world_market_price_effect : xr.DataArray
        Boolean mask for crops with a world-market price effect.

    Returns
    -------
    tuple of xr.DataArray
        New price and change in consumer surplus.
    """
    new_price = base_price.copy()

    new_price = xr.where(
        local_market_price_effect,
        (new_yield_local - demand_intercept) / demand_slope,
        new_price,
    )

    new_price = xr.where(
        world_market_price_effect,
        (new_yield_world - demand_intercept) / demand_slope,
        new_price,
    )

    change_consumer_surplus = xr.where(
        local_market_price_effect,
        -(
            (new_price - base_price) * new_yield_local
            + 0.5
            * (new_price - base_price)
            * (standard_production_sce - new_yield_local)
        ),
        0,
    )

    change_consumer_surplus = xr.where(
        world_market_price_effect,
        -(
            (new_price - base_price) * reconstructed_local_yield
            + 0.5
            * (new_price - base_price)
            * (standard_production_sce - reconstructed_local_yield)
        ),
        change_consumer_surplus,
    )

    return new_price, change_consumer_surplus
