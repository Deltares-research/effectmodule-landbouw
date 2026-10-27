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


def calculate_scenario_yields(
    standard_production_sce: xr.DataArray,
    standard_production_sce_world: xr.DataArray,
    yield_loss_sce: xr.DataArray,
    yield_loss_sce_per_crop: xr.DataArray,
    elasticity: xr.Dataset,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Calculate scenario production after yield losses.

    Parameters
    ----------
    standard_production_sce : xr.DataArray
        Standard scenario production by crop, area, and optionally year.
    standard_production_sce_world : xr.DataArray
        Standard scenario production by crop for the world market.
    yield_loss_sce : xr.DataArray
        Scenario production loss by crop, area, and year.
    yield_loss_sce_per_crop : xr.DataArray
        Scenario production loss aggregated by crop.
    elasticity : xr.Dataset
        Elasticity data containing the Dutch share of world demand.

    Returns
    -------
    tuple of xr.DataArray
        Local scenario production after losses, world scenario production
        after losses, and the Dutch share of world scenario production.
    """
    new_yield_local = standard_production_sce - yield_loss_sce

    new_yield_world_per_crop_year = (
        standard_production_sce_world - yield_loss_sce_per_crop
    )

    new_yield_local_from_world_per_crop_year = (
        elasticity["NL aandeel wereldvraag"] * new_yield_world_per_crop_year
    )

    return (
        new_yield_local,
        new_yield_world_per_crop_year,
        new_yield_local_from_world_per_crop_year,
    )


def calculate_average_standard_production(
    standard_production_local: xr.DataArray,
    standard_production_world: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate average standard production over time when time is present.

    Parameters
    ----------
    standard_production_local : xr.DataArray
        Standard production by crop and area.
    standard_production_world : xr.DataArray
        Standard production by crop for the world market.

    Returns
    -------
    tuple of xr.DataArray
        Average local and world standard production.
    """
    if "Year" in standard_production_local.dims:
        average_production_local = standard_production_local.mean("Year")

        average_production_world = standard_production_world.mean("Year")

    else:
        average_production_local = standard_production_local
        average_production_world = standard_production_world

    return (
        average_production_local,
        average_production_world,
    )
