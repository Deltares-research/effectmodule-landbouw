import xarray as xr


def NCPrijs(
    data: xr.DataArray,
    landgebruik: xr.DataArray,
    regio: xr.DataArray,
    areagrid: xr.DataArray,
    perha: bool,
):
    """
    Returns a summarized DataFrame of the input data aggregated by land use and region.

    Parameters
    ----------
    data : xr.DataArray
        The input data array to be aggregated.
    landgebruik : xr.DataArray
        The land use array corresponding to the input data.
    regio : xr.DataArray
        The region array corresponding to the input data.
    areagrid : xr.DataArray
        The area in m2 per grid cell.
    perha : bool
        Boolean indication wether "data" is expressed per hectare.
        If True, the data will be multiplied by the area in hectares per grid cell.

    Returns
    -------
    sum_df : pd.DataFrame
        The summarized DataFrame with data aggregated by land use and region.
    """

    mask_area = regio.notnull()
    data = data * mask_area
    regio = regio * mask_area

    # Convert areagrid from m2 to ha
    areagrid = areagrid * 0.0001
    if perha:
        data = data * areagrid

    regio.name = "regio"
    data.name = "data"

    database = xr.merge([data, landgebruik, regio, areagrid], compat="override")
    database = database.where(database.landgebruik != -9999)
    # database['year'] = database.data.time.dt.year

    database = database.compute()
    crop_regio_sum_da = database.groupby(["landgebruik", "regio"]).sum()

    # calculate yearlyd data from smaller timesteps
    if "time" in crop_regio_sum_da.dims:
        crop_regio_sum_da["yearly_data"] = (
            crop_regio_sum_da["data"].groupby(crop_regio_sum_da.time.dt.year).sum()
        )
        sum_df = (
            crop_regio_sum_da[["yearly_data", "areaal"]]
            .to_dataframe()
            .drop(["dx", "dy"], axis=1)
        )
        sum_df["data"] = sum_df["yearly_data"] / sum_df["areaal"]
        sum_df = sum_df.drop("yearly_data", axis=1)
    else:
        sum_df = (
            crop_regio_sum_da[["data", "areaal"]]
            .to_dataframe()
            .drop(["dx", "dy"], axis=1)
        )
        sum_df["data"] = sum_df["data"] / sum_df["areaal"]
    if "spatial_ref" in sum_df.columns:
        sum_df = sum_df.drop("spatial_ref", axis=1)
    return sum_df
