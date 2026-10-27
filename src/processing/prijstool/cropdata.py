import pandas as pd
import xarray as xr

from processing.prijstool.aggregation import NCPrijs


def create_cropdata(
    potASC_combi,
    act_dry_ASC_combi,
    Ylderv_med_combi,
    arbdASC_sce,
    enrgASC_sce,
    landgebruik_sce,
    landgebruik_ref,
    regios_ar,
    areaalasc_sce,
    areaalasc_ref,
    years,
):
    """
    Create the crop database used as input for the Prijstool.

    Parameters
    ----------
    potASC_combi : xr.DataArray
        Potential yield.
    act_dry_ASC_combi : xr.DataArray
        Actual yield.
    Ylderv_med_combi : xr.DataArray
        Reference median yield loss.
    arbdASC_sce : xr.DataArray
        Labour costs.
    enrgASC_sce : xr.DataArray
        Energy costs.
    landgebruik_sce : xr.DataArray
        Scenario land use.
    landgebruik_ref : xr.DataArray
        Reference land use.
    regios_ar : xr.DataArray
        Rasterized regions.
    areaalasc_sce : xr.DataArray
        Scenario cell area.
    areaalasc_ref : xr.DataArray
        Reference cell area.
    years : np.ndarray
        Years to include.

    Returns
    -------
    pd.DataFrame
        Crop database.
    """

    # Load Yield
    # MvdV 191128 added switch perha to muliply with ha per cell or not Tru for yields, Slase for costs
    pot_opbrengst = NCPrijs(
        potASC_combi[potASC_combi.time.dt.year.isin(years)],
        landgebruik_sce,
        regios_ar,
        areaalasc_sce,
        perha=True,
    ).rename(
        columns={
            "data": "pot_opbrengst",
            "Areagrid": "areaal",
        }
    )

    act_opbrengst = (
        NCPrijs(
            act_dry_ASC_combi[act_dry_ASC_combi.time.dt.year.isin(years)],
            landgebruik_sce,
            regios_ar,
            areaalasc_sce,
            perha=True,
        )
        .drop(columns=["areaal"])
        .rename(columns={"data": "act_opbrengst"})
    )

    ref_med_derving = (
        NCPrijs(
            Ylderv_med_combi,
            landgebruik_ref,
            regios_ar,
            areaalasc_ref,
            perha=True,
        )
        .drop(columns=["areaal"])
        .rename(columns={"data": "ref_med_derv"})
    )

    # Load Labor costs
    # MvdV 191128 added switch perha to muliply with ha per cell or not Tru for yields, Slase for costs
    arbeidskosten = (
        NCPrijs(
            arbdASC_sce[arbdASC_sce.time.dt.year.isin(years)],
            landgebruik_sce,
            regios_ar,
            areaalasc_sce,
            perha=False,
        )
        .drop(columns=["areaal"])
        .rename(columns={"data": "arbeidskosten"})
    )

    # Load Energy costs
    energiekosten = (
        NCPrijs(
            enrgASC_sce[enrgASC_sce.time.dt.year.isin(years)],
            landgebruik_sce,
            regios_ar,
            areaalasc_sce,
            perha=False,
        )
        .drop(columns=["areaal"])
        .rename(columns={"data": "energiekosten"})
    )

    # Merge Potential and Actual Yield and make Crop Database
    cropdata = act_opbrengst.merge(
        pot_opbrengst,
        left_index=True,
        right_index=True,
    )

    cropdata = cropdata.merge(
        ref_med_derving,
        left_index=True,
        right_index=True,
    )

    cropdata = cropdata.merge(
        arbeidskosten,
        left_index=True,
        right_index=True,
    )

    cropdata = cropdata.merge(
        energiekosten,
        left_index=True,
        right_index=True,
    )

    return cropdata


def create_cropdata_xr(
    cropdata: pd.DataFrame,
    crop_prices: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> xr.Dataset:
    """
    Convert crop data to an xarray Dataset and add missing regions.

    Crop numbers are mapped to crop names using the elasticity data.
    Regions present in the crop-price data but absent from the crop
    data are added and filled with zeros.

    Parameters
    ----------
    cropdata : pd.DataFrame
        Crop data containing the columns 'Crop', 'Area', and 'Year'.
    crop_prices : pd.DataFrame
        Crop-price data containing the 'regio_nummer' column.
    elasticity : pd.DataFrame
        Elasticity data containing the mapping between crop numbers
        and crop names.

    Returns
    -------
    xr.Dataset
        Crop data indexed by Crop, Area, and Year, including missing
        regions filled with zeros.
    """
    crop_name_num = pd.Series(
        elasticity.columns,
        index=elasticity.loc["nummer"],
    )

    cropdata["Cropnum"] = cropdata["Crop"]
    cropdata["Crop"] = cropdata["Crop"].map(crop_name_num)

    cropdata = cropdata.set_index(["Crop", "Area", "Year"])

    not_present_regions = crop_prices.regio_nummer[
        ~crop_prices.regio_nummer.isin(cropdata.index.get_level_values("Area").unique())
    ]

    for region in not_present_regions:
        add_region = pd.MultiIndex.from_product(
            [
                cropdata.index.get_level_values("Crop").unique(),
                [region],
                cropdata.index.get_level_values("Year").unique(),
            ],
            names=cropdata.index.names,
        )

        add_region_df = pd.DataFrame(
            0.0,
            index=add_region,
            columns=cropdata.columns,
        )

        cropdata = pd.concat([cropdata, add_region_df]).sort_index()

    return cropdata.to_xarray().fillna(0)


def finalise_cropdata(cropdata: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Finalize the crop data by adding the scenario and resetting the index.

    Parameters
    ----------
    cropdata : pd.DataFrame
        Crop database.
    scenario : str
        Scenario name.

    Returns
    -------
    pd.DataFrame
        Finalised crop database.
    """
    cropdata = cropdata.copy()

    cropdata["Scenario"] = scenario

    cropdata.reset_index(inplace=True)

    cropdata = cropdata[
        [
            "regio",
            "landgebruik",
            "act_opbrengst",
            "pot_opbrengst",
            "areaal",
            "damage_calculation_yield",
            "derv_opbrengst",
            "ref_med_derv",
            "arbeidskosten",
            "energiekosten",
            "year",
            "Scenario",
        ]
    ]

    cropdata = cropdata.rename(columns={"year": "Year"})

    return cropdata
