import xarray as xr


def calculate_production_and_costs(
    cropdata_sce: xr.Dataset,
    cropdata_ref: xr.Dataset,
    standard_yield_sce: xr.DataArray,
    standard_yield_ref: xr.DataArray,
    multpotyield: float,
    multredyield: float,
):
    """
    Calculate standard production, production loss, and costs.

    Parameters
    ----------
    cropdata_sce : xr.Dataset
        Crop data for the scenario.
    cropdata_ref : xr.Dataset
        Crop data for the reference situation.
    standard_yield_sce : xr.DataArray
        Standard yield for the scenario.
    standard_yield_ref : xr.DataArray
        Standard yield for the reference situation.
    multpotyield : float
        Multiplication factor applied to standard production.
    multredyield : float
        Multiplication factor applied to yield reduction.

    Returns
    -------
    tuple
        Scenario area, reference area, reference standard production,
        scenario standard production, scenario production loss,
        scenario labour costs, and scenario energy costs.
    """
    areaal_sce = cropdata_sce["areaal"].mean("Year")
    areaal_ref = cropdata_ref["areaal"].mean("Year")

    ############
    # AREAAL BIJ Y IS NU AREAAL VAN SCENARIO EN NIET VAN DE REF
    # EN MOET DE K6 DUS VAN DE REF BLIJVEN OF PER SCENARIO?
    # DUS DIT ,MOET IN DE BASIS OOK AL EEN YREF EN YSCE ZIJN
    # INCLUSIEF AREAAL VOOR YREF EN YSCE

    standard_opbrengst_kg_ref = areaal_ref * standard_yield_ref * multpotyield

    standard_opbrengst_kg_sce = areaal_sce * standard_yield_sce * multpotyield

    droogtederving_kg_sce = areaal_sce * cropdata_sce["derv_opbrengst"] * multredyield

    arbeidskosten_sce = areaal_sce * cropdata_sce["arbeidskosten"]

    energiekosten_sce = areaal_sce * cropdata_sce["energiekosten"]

    return (
        areaal_sce,
        standard_opbrengst_kg_ref,
        standard_opbrengst_kg_sce,
        droogtederving_kg_sce,
        arbeidskosten_sce,
        energiekosten_sce,
    )


def aggregate_production_by_crop(
    standard_production_ref: xr.DataArray,
    standard_production_sce: xr.DataArray,
    yield_loss_sce: xr.DataArray,
    elasticity: xr.Dataset,
) -> tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]:
    """
    Aggregate production and yield loss by crop.

    Standard production is summed over all areas. The resulting
    reference and scenario production are converted to world
    production using the Dutch share of world supply.

    Parameters
    ----------
    standard_production_ref : xr.DataArray
        Standard production for the reference situation.
    standard_production_sce : xr.DataArray
        Standard production for the scenario.
    yield_loss_sce : xr.DataArray
        Scenario yield loss.
    elasticity : xr.Dataset
        Elasticity data containing the Dutch share of world supply.

    Returns
    -------
    tuple of xr.DataArray
        Reference world production by crop, scenario world production
        by crop, and scenario yield loss by crop.
    """

    # do sums over area
    # DIT MOETEN EEN YREF EN EEN YSCE VARIANT WORDEN
    #############################

    production_ref_per_crop = standard_production_ref.sum("Area")
    production_sce_per_crop = standard_production_sce.sum("Area")

    production_ref_per_crop_world = (
        production_ref_per_crop / elasticity["NL aandeel wereldaanbod"]
    )

    production_sce_per_crop_world = (
        production_sce_per_crop / elasticity["NL aandeel wereldaanbod"]
    )

    yield_loss_sce_per_crop = yield_loss_sce.sum("Area")

    return (
        production_ref_per_crop_world,
        production_sce_per_crop_world,
        yield_loss_sce_per_crop,
    )
