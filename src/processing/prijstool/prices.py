import pandas as pd
import xarray as xr


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
