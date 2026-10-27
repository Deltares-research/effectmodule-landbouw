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
        areaal_ref,
        standard_opbrengst_kg_ref,
        standard_opbrengst_kg_sce,
        droogtederving_kg_sce,
        arbeidskosten_sce,
        energiekosten_sce,
    )
