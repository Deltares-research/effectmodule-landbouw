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
