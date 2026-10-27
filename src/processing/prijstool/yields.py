import xarray as xr


def get_standard_yield(
    cropdata: xr.Dataset,
    method: str,
) -> xr.DataArray:
    """
    Select the standard yield based on the calculation method.

    Parameters
    ----------
    cropdata : xr.Dataset
        Crop data containing the yield variables.
    method : str
        Yield-loss calculation method.

    Returns
    -------
    xr.DataArray
        Standard yield used in the Prijstool calculation.
    """
    if method == "optdervActPot":
        standard_yield = cropdata["pot_opbrengst"]

    elif method == "optdervAct_Actgem":
        standard_yield = cropdata["gem_opbrengst"]

    elif method in [
        "optdervAct_Actmed",
        "optdervderv_ActMed",
        "optdervderv_ActMed_new",
    ]:
        standard_yield = cropdata["act_opbrengst"].median("Year")

    else:
        raise ValueError(f"Unknown calculation method: {method}")

    return standard_yield
