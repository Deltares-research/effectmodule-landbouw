"""
Met dit script worden de volgende zaken berekend en als idf en figuur weggeschreven:
    - Referentie gewasopbrengst (langjarig en per jaar)
    - Actuele gewasopbrengst (langjarig en per jaar)
    - Schadepercentages voor verschillende schadetermen (langjarig en per jaar)
    - Scades in kg/ha voor verschillende schadetermen  (langjarig en per jaar)

Hierbij een korte uitleg van deze berekeningen

1.	Bouwland, exclusief mais.
    De volgende stappen eerst per jaar.
    -	De referentieopbrengst, zijnde:   wof_Wstorg_organs_yld_ind_31dec =
            wof_Wstorg_organs_yld_act_31dec/ wof_Wstorg_organs_yldrel_31dec
    -	De schadetermen in kg/ha , adhv schadepercentages en referentieopbrengst      :
        o	Totale schade = wof_Wstorg_organs_yld_dam_kg_31dec           =  (1/100)*
            wof_Wstorg_organs_yld_ind_31dec* wof_Wstorg_organs_yld_dam_31dec
        o	Indirecte schade = wof_Wstorg_organs_yld_dam_ind_kg_31dec = (1/100)*
            wof_Wstorg_organs_yld_ind_31dec* wof_Wstorg_organs_yld_dam_ind_31dec
        o	Droogteschade = wof_Wstorg_organs_yld_dam_dry_kg_31dec = (1/100)*
            wof_Wstorg_organs_yld_ind_31dec* wof_Wstorg_organs_yld_dam_dry_31dec
        o	Natschade = wof_Wstorg_organs_yld_dam_wet_kg_31dec       = (1/100)*
            wof_Wstorg_organs_yld_ind_31dec* wof_Wstorg_organs_yld_dam_wet_31dec
        o	Zoutschade = wof_Wstorg_organs_yld_dam_salt_kg_31dec      = (1/100)* wof_Wstorg_organs_yld_ind_31dec* wof_Wstorg_organs_yld_dam_salt_31dec
    Voor het langjarig gemiddelde worden vervolgens:
    -	De referentieopbrengsten gesommeerd
    -	De schades van de deeltermen gesommeerd
    -	De som schades gedeeld door de som referentieopbrengst, maal 100*

2.	Mais
    Bij mais worden bovenstaande stappen ook voor leaves en stems gedaan.  Eerst de stappen per jaar, voor organs en leaves helemaal apart, alles in kg/ha

    Bij het berekenen van het langjarig gemiddelde worden de referentie-opbrengsten van organs en leaves eerst gesommeerd. Vervolgens per schade term de schades van organs en leaves gesommeerd. Pas in de laatste stap voor de 5 schadetermen de som schades gedeelde door som opbrengsten, maal 100%.  Ook interessant is het langjarig gemiddelde van de referentieopbrengst en van de actuele opbrengst.

3.	Gras
    Als bouwland, maar dan met stems_leaves, en met afwijkende peil datum (31 okt, want daarna is er nauwelijks meer voedingswaarde aan gras)
    Bij gras wordt niet alleen het gemaaide gras op 31 oktober beschouwd, maar ook het maaibare gras dat nog op het land staat op 31 okt.

    Het maaibare gras is daarbij het gras op het land minus de stoppelgrootte van gras (want de stoppel wordt niet gemaaid. )
    Deze correctie op de uitvoer is best complex:
        ACTUELE EN POTENTIELE OPBRENGST CORRIGEREN
        - Het maaibare deel is het deel op het land (_act en _pot) minus de stoppelgrootte.
        - Als dat negatief is, dan toch 0 aannemen.
        - het maaibare deel optellen bij de yld_act en de yld_pot.
        - MAAR de yld_pot is geen directe uitvoer van WOFOST. deze dus eerst berekenen als yld_pot = yld_act / yld_rel
        CORRECTIE DOORVOEREN OP RELATIEVE OPBRENGST EN SCHADEPERCENTAGES
        - Opnieuw de yld_rel en yld_dam bepalen aan de hand van de gecorrigeerde yld_act en yld_pot
        - De verandering van de schade (yld_dam) bepalen
        - En dit verschil ook doorvoeren voor de uitgesplitste schades (droog, nat, zout, indirect)


"""

import copy
import os

import geopandas as gpd
import imod
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path

import imod
import xarray as xr

WOFOST_COMPONENTS = ["stems_leaves", "storg_organs"]

WOFOST_VARIABLES = [
    "ind",
    "act",
    "pot",
    "yld_act",
    "yld_dam",
    "yld_dam_dry",
    "yld_dam_ind",
    "yld_dam_salt",
    "yld_dam_wet",
]

DAMAGE_TERMS = ["dam", "dam_dry", "dam_ind", "dam_salt", "dam_wet"]
EXTENDED_DAMAGE_TERMS = DAMAGE_TERMS + ["schade_direct", "schade_nat"]

GRASS_STUBBLE_KG_HA = (
    700  # kg droge stof per ha, komt uit wofost .crp file, de TAGPREST variabele
)

LGN_GRAS = [1]
LGN_MAIS = [2]
LGN_BOUWLAND = [3, 4, 5]  # , 6] 6 is overig en wordt met FAO berekend


def load_wofost_data(run_dir: Path, startjaar: int, eindjaar: int) -> dict:
    """
    Load WOFOST model output for all required variables and crop components.

    Parameters
    ----------
    run_dir : Path
        Directory containing the MetaSWAP/WOFOST model output.
    startjaar : int
        First year to load.
    eindjaar : int
        Last year to load.

    Returns
    -------
    dict
        Nested dictionary containing WOFOST output by variable
        and crop component.
    """

    wofost_data = {}
    for variable in WOFOST_VARIABLES:
        wofost_data[variable] = {}

        for component in WOFOST_COMPONENTS:
            folder = run_dir / "metaswap" / f"wof_W{component}_{variable}"

            files = []

            for year in range(startjaar, eindjaar + 1):
                files.extend(
                    sorted(
                        folder.glob(f"wof_W{component}_{variable}_{year}????_l1.idf")
                    )
                )

            data = imod.idf.open(files).sel(layer=1).drop("layer")

            wofost_data[variable][component] = data

    return wofost_data


def select_annual_outputs(
    wofost_data: dict,
) -> dict:
    """
    Select annual WOFOST outputs for the relevant crop groups.

    Grass uses values from 31 October.
    Arable crops and maize use values from 31 December.

    Parameters
    ----------
    wofost_data : dict
        Nested dictionary containing WOFOST output by variable
        and crop component.

    Returns
    -------
    dict
        Nested dictionary containing annual WOFOST output for
        grass and for arable crops/maize.
    """
    annual_outputs = {}

    for variable in WOFOST_VARIABLES:
        annual_outputs[variable] = {}

        for component in WOFOST_COMPONENTS:
            data = wofost_data[variable][component]

            annual_outputs[variable][component] = {
                "grass": data[(data["time.month"] == 10) & (data["time.day"] == 31)],
                "other_crops": data[
                    (data["time.month"] == 12) & (data["time.day"] == 31)
                ],
            }

    return annual_outputs


def correct_grass_yield(
    annual_outputs: dict,
    stubble_yield: float = 700,
    damage_terms: list[str] = DAMAGE_TERMS,
) -> dict:
    """
    Correct WOFOST grass yield for harvestable biomass on 31 October.

    The harvestable grass remaining in the field is calculated by
    subtracting the grass stubble yield from the actual and potential
    stems-and-leaves biomass. Negative harvestable biomass is set to
    zero.

    The harvestable biomass is added to actual and potential yield.
    Relative yield and total damage are then recalculated, and the
    individual damage components are adjusted proportionally.

    Parameters
    ----------
    annual_outputs : dict
        Nested dictionary containing annual WOFOST output by variable,
        component, and crop group.
    stubble_yield : float, default 700
        Grass stubble yield in kg dry matter per hectare.
    damage_terms : list[str]
        Damage terms whose percentages should be corrected.

    Returns
    -------
    dict
        Annual WOFOST outputs with corrected grass yield and damage.
    """
    component = "stems_leaves"
    crop_group = "grass"

    harvestable_actual = annual_outputs["act"][component][crop_group] - stubble_yield

    harvestable_potential = annual_outputs["pot"][component][crop_group] - stubble_yield

    # Negative harvestable biomass is set to zero.
    harvestable_actual = harvestable_actual.where(
        harvestable_actual > 0,
        0,
    )

    harvestable_potential = harvestable_potential.where(
        harvestable_actual > 0,
        0,
    )

    # Derive potential yield because it is not supplied directly
    # as a WOFOST yield output.
    potential_yield = annual_outputs["yld_act"][component][crop_group] / (
        1 - annual_outputs["yld_dam"][component][crop_group] / 100
    )

    # Add harvestable biomass to actual and potential yield.
    annual_outputs["yld_act"][component][crop_group] = (
        annual_outputs["yld_act"][component][crop_group] + harvestable_actual
    )

    potential_yield = potential_yield + harvestable_potential

    # Recalculate relative yield.
    relative_yield = annual_outputs["yld_act"][component][crop_group] / potential_yield

    # Store corrected potential/reference yield.
    annual_outputs["ind"][component][crop_group] = potential_yield

    # Preserve the old total damage percentage.
    old_total_damage = annual_outputs["yld_dam"][component][crop_group].copy()

    # Recalculate total damage percentage.
    annual_outputs["yld_dam"][component][crop_group] = (1 - relative_yield) * 100

    damage_adjustment_factor = (
        annual_outputs["yld_dam"][component][crop_group] / old_total_damage
    )

    # Adjust the individual damage percentages proportionally.
    for damage_term in damage_terms:
        if damage_term == "dam":
            continue

        variable = f"yld_{damage_term}"

        annual_outputs[variable][component][crop_group] = (
            annual_outputs[variable][component][crop_group] * damage_adjustment_factor
        )

    return annual_outputs


def set_annual_maximum_yields(
    wofost_data: dict,
    annual_outputs: dict,
    start_year: int,
    end_year: int,
) -> dict:
    """
    Replace year-end actual and reference yields with annual maxima.

    For each year and WOFOST crop component, the maximum value of
    'act' and 'ind' over the complete year is assigned to the
    non-grass annual output.

    Grass is not modified because grass uses the 31 October output.

    Parameters
    ----------
    wofost_data : dict
        Full WOFOST time series by variable and crop component.
    annual_outputs : dict
        Annual WOFOST outputs created by select_annual_outputs().
    start_year : int
        First year to process.
    end_year : int
        Last year to process.

    Returns
    -------
    dict
        Annual WOFOST outputs with annual maxima assigned to
        'act' and 'ind' for non-grass crops.
    """
    for year in range(start_year, end_year + 1):
        for component in WOFOST_COMPONENTS:
            data_act = wofost_data["act"][component][
                wofost_data["act"][component]["time.year"] == year
            ]

            data_ind = wofost_data["ind"][component][
                wofost_data["ind"][component]["time.year"] == year
            ]

            annual_outputs["act"][component]["other_crops"][
                annual_outputs["act"][component]["other_crops"]["time.year"] == year
            ] = data_act.max(dim="time")

            annual_outputs["ind"][component]["other_crops"][
                annual_outputs["ind"][component]["other_crops"]["time.year"] == year
            ] = data_ind.max(dim="time")

    return annual_outputs


def add_direct_and_wet_damage(
    annual_outputs: dict,
) -> dict:
    """
    Add derived direct and wet damage percentages.

    Direct damage is the sum of drought, wet, and salt damage.
    Wet-related damage is the sum of indirect and wet damage.

    Parameters
    ----------
    annual_outputs : dict
        Annual WOFOST outputs by variable, crop component,
        and crop group.

    Returns
    -------
    dict
        Annual outputs including the derived damage terms.
    """
    annual_outputs["yld_schade_direct"] = {}
    annual_outputs["yld_schade_nat"] = {}

    for component in WOFOST_COMPONENTS:
        annual_outputs["yld_schade_direct"][component] = {}
        annual_outputs["yld_schade_nat"][component] = {}

        for crop_group in ["grass", "other_crops"]:
            annual_outputs["yld_schade_direct"][component][crop_group] = (
                annual_outputs["yld_dam_dry"][component][crop_group]
                + annual_outputs["yld_dam_wet"][component][crop_group]
                + annual_outputs["yld_dam_salt"][component][crop_group]
            )

            annual_outputs["yld_schade_nat"][component][crop_group] = (
                annual_outputs["yld_dam_ind"][component][crop_group]
                + annual_outputs["yld_dam_wet"][component][crop_group]
            )

    return annual_outputs


def calculate_damage_kg_ha(
    reference_yield: xr.DataArray,
    damage_percentage: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate crop damage in kg/ha.

    Parameters
    ----------
    reference_yield : xr.DataArray
        Reference crop yield in kg/ha.
    damage_percentage : xr.DataArray
        Crop damage as a percentage of reference yield.

    Returns
    -------
    xr.DataArray
        Crop damage in kg/ha.
    """
    damage_kg_ha = reference_yield * damage_percentage / 100

    return damage_kg_ha


def calculate_all_damage_kg_ha(
    annual_outputs: dict,
    damage_terms: list[str],
) -> dict:
    """
    Calculate all WOFOST damage terms in kg/ha.

    Damage is calculated for every damage term, WOFOST crop
    component, and crop group.

    Parameters
    ----------
    annual_outputs : dict
        Annual WOFOST outputs containing reference yield and
        damage percentages.
    damage_terms : list[str]
        Damage terms to convert from percentages to kg/ha.

    Returns
    -------
    dict
        Nested dictionary containing damage in kg/ha by
        damage term, WOFOST component, and crop group.
    """
    damage_kg_ha = {}

    for damage_term in damage_terms:
        damage_kg_ha[damage_term] = {}

        for component in WOFOST_COMPONENTS:
            damage_kg_ha[damage_term][component] = {}

            for crop_group in [
                "grass",
                "other_crops",
            ]:
                reference_yield = annual_outputs["ind"][component][crop_group]

                damage_percentage = annual_outputs[f"yld_{damage_term}"][component][
                    crop_group
                ]

                damage_kg_ha[damage_term][component][crop_group] = (
                    calculate_damage_kg_ha(
                        reference_yield=reference_yield,
                        damage_percentage=damage_percentage,
                    )
                )

    return damage_kg_ha


def sum_outputs_over_time(
    annual_outputs: dict,
    damage_kg_ha: dict,
    damage_terms: list[str],
) -> dict:
    """
    Sum annual WOFOST yield and damage outputs over time.

    Parameters
    ----------
    annual_outputs : dict
        Annual WOFOST outputs by variable, component, and crop group.
    damage_kg_ha : dict
        Damage in kg/ha by damage term, component, and crop group.
    damage_terms : list[str]
        Damage terms to aggregate.

    Returns
    -------
    dict
        Time-summed reference yield, actual yield, WOFOST actual
        biomass, and damage in kg/ha.
    """
    summed_outputs = {
        "reference_yield": {},
        "actual_yield": {},
        "actual_biomass": {},
        "damage_kg_ha": {},
    }

    for component in WOFOST_COMPONENTS:
        summed_outputs["reference_yield"][component] = {}
        summed_outputs["actual_yield"][component] = {}
        summed_outputs["actual_biomass"][component] = {}

        for crop_group in ["grass", "other_crops"]:
            summed_outputs["reference_yield"][component][crop_group] = annual_outputs[
                "ind"
            ][component][crop_group].sum("time")

            summed_outputs["actual_yield"][component][crop_group] = annual_outputs[
                "yld_act"
            ][component][crop_group].sum("time")

            summed_outputs["actual_biomass"][component][crop_group] = annual_outputs[
                "act"
            ][component][crop_group].sum("time")

    for damage_term in damage_terms:
        summed_outputs["damage_kg_ha"][damage_term] = {}

        for component in WOFOST_COMPONENTS:
            summed_outputs["damage_kg_ha"][damage_term][component] = {}

            for crop_group in ["grass", "other_crops"]:
                summed_outputs["damage_kg_ha"][damage_term][component][crop_group] = (
                    damage_kg_ha[damage_term][component][crop_group].sum("time")
                )

    return summed_outputs


def combine_outputs_by_crop(
    outputs: dict,
    lgn: xr.DataArray,
    damage_terms: list[str],
) -> dict:
    """
    Combine WOFOST component outputs into crop-specific raster outputs.

    Arable crops use storage-organ output, maize uses the sum of
    storage organs and stems/leaves, and grass uses stems/leaves.
    Crop-specific outputs are combined using the LGN crop classes.

    Parameters
    ----------
    summed_outputs : dict
        Time-summed WOFOST outputs by component and crop group.
    lgn : xr.DataArray
        LGN land-use raster aligned with the WOFOST output grid.
    damage_terms : list[str]
        Damage terms to combine by crop.

    Returns
    -------
    dict
        Combined reference yield, actual yield, and damage in kg/ha.
    """
    combined_outputs = {
        "reference_yield": {},
        "actual_yield": {},
        "damage_kg_ha": {},
    }

    # Reference yield
    combined_outputs["reference_yield"]["arable"] = outputs["reference_yield"][
        "storg_organs"
    ]["other_crops"]

    combined_outputs["reference_yield"]["maize"] = (
        outputs["reference_yield"]["storg_organs"]["other_crops"]
        + outputs["reference_yield"]["stems_leaves"]["other_crops"]
    )

    combined_outputs["reference_yield"]["grass"] = outputs["reference_yield"][
        "stems_leaves"
    ]["grass"]

    combined_outputs["reference_yield"]["total"] = (
        combined_outputs["reference_yield"]["arable"].where(
            lgn.isin(LGN_BOUWLAND),
            0,
        )
        + combined_outputs["reference_yield"]["maize"].where(
            lgn.isin(LGN_MAIS),
            0,
        )
        + combined_outputs["reference_yield"]["grass"].where(
            lgn.isin(LGN_GRAS),
            0,
        )
    )

    # Actual yield
    combined_outputs["actual_yield"]["arable"] = outputs["actual_biomass"][
        "storg_organs"
    ]["other_crops"]

    combined_outputs["actual_yield"]["maize"] = (
        outputs["actual_biomass"]["storg_organs"]["other_crops"]
        + outputs["actual_biomass"]["stems_leaves"]["other_crops"]
    )

    # Grass deliberately uses yld_act instead of act.
    combined_outputs["actual_yield"]["grass"] = outputs["actual_yield"]["stems_leaves"][
        "grass"
    ]

    combined_outputs["actual_yield"]["total"] = (
        combined_outputs["actual_yield"]["arable"].where(
            lgn.isin(LGN_BOUWLAND),
            0,
        )
        + combined_outputs["actual_yield"]["maize"].where(
            lgn.isin(LGN_MAIS),
            0,
        )
        + combined_outputs["actual_yield"]["grass"].where(
            lgn.isin(LGN_GRAS),
            0,
        )
    )

    # Damage
    for damage_term in damage_terms:
        combined_outputs["damage_kg_ha"][damage_term] = {}

        combined_outputs["damage_kg_ha"][damage_term]["arable"] = outputs[
            "damage_kg_ha"
        ][damage_term]["storg_organs"]["other_crops"]

        combined_outputs["damage_kg_ha"][damage_term]["maize"] = (
            outputs["damage_kg_ha"][damage_term]["storg_organs"]["other_crops"]
            + outputs["damage_kg_ha"][damage_term]["stems_leaves"]["other_crops"]
        )

        combined_outputs["damage_kg_ha"][damage_term]["grass"] = outputs[
            "damage_kg_ha"
        ][damage_term]["stems_leaves"]["grass"]

        combined_outputs["damage_kg_ha"][damage_term]["total"] = (
            combined_outputs["damage_kg_ha"][damage_term]["arable"].where(
                lgn.isin(LGN_BOUWLAND),
                0,
            )
            + combined_outputs["damage_kg_ha"][damage_term]["maize"].where(
                lgn.isin(LGN_MAIS),
                0,
            )
            + combined_outputs["damage_kg_ha"][damage_term]["grass"].where(
                lgn.isin(LGN_GRAS),
                0,
            )
        )

    return combined_outputs


def assign_year_coordinate(
    data: xr.DataArray,
) -> xr.DataArray:
    """
    Replace the datetime coordinate with integer calendar years.

    Parameters
    ----------
    data : xr.DataArray
        Data with a datetime-based time coordinate.

    Returns
    -------
    xr.DataArray
        Data with the time coordinate expressed as integer years.
    """
    return data.assign_coords(time=data.time.dt.year)


def prepare_annual_outputs_by_year(
    annual_outputs: dict,
    damage_kg_ha: dict,
    damage_terms: list[str],
) -> dict:
    """
    Prepare annual WOFOST outputs for crop combination.

    October and December timestamps are replaced with calendar years
    so grass and other-crop outputs can be aligned and combined.

    Parameters
    ----------
    annual_outputs : dict
        Annual WOFOST outputs by variable, component, and crop group.
    damage_kg_ha : dict
        Annual damage in kg/ha by damage term, component, and crop group.
    damage_terms : list[str]
        Damage terms to prepare.

    Returns
    -------
    dict
        Annual reference yield, actual yield, actual biomass, and
        damage, retaining the time dimension as integer years.
    """
    yearly_outputs = {
        "reference_yield": {},
        "actual_yield": {},
        "actual_biomass": {},
        "damage_kg_ha": {},
    }

    for component in WOFOST_COMPONENTS:
        yearly_outputs["reference_yield"][component] = {}
        yearly_outputs["actual_yield"][component] = {}
        yearly_outputs["actual_biomass"][component] = {}

        for crop_group in ["grass", "other_crops"]:
            yearly_outputs["reference_yield"][component][crop_group] = (
                assign_year_coordinate(annual_outputs["ind"][component][crop_group])
            )

            yearly_outputs["actual_yield"][component][crop_group] = (
                assign_year_coordinate(annual_outputs["yld_act"][component][crop_group])
            )

            yearly_outputs["actual_biomass"][component][crop_group] = (
                assign_year_coordinate(annual_outputs["act"][component][crop_group])
            )

    for damage_term in damage_terms:
        yearly_outputs["damage_kg_ha"][damage_term] = {}

        for component in WOFOST_COMPONENTS:
            yearly_outputs["damage_kg_ha"][damage_term][component] = {}

            for crop_group in ["grass", "other_crops"]:
                yearly_outputs["damage_kg_ha"][damage_term][component][crop_group] = (
                    assign_year_coordinate(
                        damage_kg_ha[damage_term][component][crop_group]
                    )
                )

    return yearly_outputs


def calculate_damage_fractions(
    combined_outputs: dict,
    damage_terms: list[str],
) -> dict:
    """
    Calculate crop damage as a fraction of reference yield.

    Parameters
    ----------
    combined_outputs : dict
        Combined WOFOST crop outputs containing reference yield
        and damage in kg/ha.
    damage_terms : list[str]
        Damage terms to calculate.

    Returns
    -------
    dict
        Damage fraction for each damage term.
    """
    damage_fractions = {}

    reference_yield = combined_outputs["reference_yield"]["total"]

    for damage_term in damage_terms:
        damage_fractions[damage_term] = (
            combined_outputs["damage_kg_ha"][damage_term]["total"] / reference_yield
        )

    return damage_fractions
