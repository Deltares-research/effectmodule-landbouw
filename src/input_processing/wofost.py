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


if __name__ == "__main__":
    #################################
    ## IN TE VULLEN DOOR GEBRUIKER ##
    #################################

    run_dirs = [rf"e:\LHM_4.3.0b\preprocessed_runs\gekalibreerd_2009_2018"]
    startjaar = 2011
    eindjaar = 2018

    # %%

    ################
    ## PARAMETERS ##
    ################

    path_lgn_kaart = r"e:\LHM_4.1.1\Data\metaswap\grid\LGN250.IDF"
    ## Parameters
    wofost_onderdelen = ["stems_leaves", "storg_organs"]
    wofost_variabelen = [
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
    wofost_schadetermen = ["dam", "dam_dry", "dam_ind", "dam_salt", "dam_wet"]
    schadetermen_uitgebreid = wofost_schadetermen + ["schade_direct", "schade_nat"]
    datums = ["31dec", "31okt"]  ## Voor gras is 31dec nodig, voor andere gewassen 31okt

    LGN_gras = [1]
    LGN_mais = [2]
    LGN_bouwland = [3, 4, 5]  # , 6] 6 is overig en wordt met FAO berekend
    dvsharv = 1.75

    ## LGN kaart inlezen
    lgn = imod.idf.open(path_lgn_kaart)

    ## Plaatjes maken voorbereiden
    loc_leg_fractie = (
        r"e:\LHM_master\Data\3_Basisdata_postprocessing\legendas\droogtestress.leg"
    )
    loc_leg_gewasopbrengst = r"e:\LHM_master\Data\3_Basisdata_postprocessing\legendas\gewasopbrengst_kgha.leg"
    loc_leg_gewasschade = (
        r"e:\LHM_master\Data\3_Basisdata_postprocessing\legendas\gewasschade_kgha.leg"
    )
    loc_shp_provincies = r"e:\LHM_master\Data\3_Basisdata_postprocessing\shp_provincies\2021_provincies_zonder_water.shp"

    aantal_jaar = eindjaar - startjaar + 1

    stoppelgrootte_gras = (
        700  # kg droge stof per ha, komt uit wofost .crp file, de TAGPREST variabele
    )

    ###################
    ## VOORBEREIDING ##
    ###################

    ## Legenda's en overlays voorbereiden voor figuren
    leg_fractie = imod.visualize.spatial.read_imod_legend(loc_leg_fractie)
    leg_opbrengst = imod.visualize.spatial.read_imod_legend(loc_leg_gewasopbrengst)
    leg_schade = imod.visualize.spatial.read_imod_legend(loc_leg_gewasschade)
    provincies = gpd.read_file(loc_shp_provincies)
    overlays = [
        {"gdf": provincies, "color": "none", "edgecolor": "black", "linewidth": 0.5}
    ]

    for r in run_dirs:

        ## Per modelrun de folders maken
        dirs = [
            rf"{r}/postprocessing_{startjaar}_{eindjaar}",
            rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade",
            rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade_hist",
        ]

        for dr in dirs:
            if not os.path.exists(dr):
                os.makedirs(dr)

        ## Dictionaries aanmaken om te vullen, voor alle modeluitvoer en cumulatieve modeluitvoer per jaar
        wof_modeluitvoer = dict.fromkeys(wofost_variabelen)
        wof_modeluitvoer_cum = dict.fromkeys(wofost_variabelen)

        ## Door wofost uitvoer heenlopen
        for var in wofost_variabelen:
            wof_modeluitvoer[var] = dict.fromkeys(wofost_onderdelen)
            wof_modeluitvoer_cum[var] = dict.fromkeys(wofost_onderdelen)
            for onderdeel in wofost_onderdelen:
                ## Dictionaries maken
                wof_modeluitvoer[var][onderdeel] = {}
                wof_modeluitvoer_cum[var][onderdeel] = {}
                ## Modeluitvoer inlezen
                wof_modeluitvoer[var][onderdeel] = (
                    imod.idf.open(
                        r
                        + rf"/metaswap/wof_W"
                        + onderdeel
                        + "_"
                        + var
                        + "/wof_W"
                        + onderdeel
                        + "_"
                        + var
                        + "_????????_l1.idf"
                    )
                    .sel(time=slice(f"{startjaar}-01-01", f"{eindjaar}-12-31"))
                    .sel(layer=1)
                    .drop("layer")
                )
                # dit stuk hieronder gaat naar de functie select_annual_outputs()
                ## 31 december en 31 oktober selecteren
                wof_modeluitvoer_cum[var][onderdeel]["31dec"] = wof_modeluitvoer[var][
                    onderdeel
                ][
                    (wof_modeluitvoer[var][onderdeel]["time.month"] == 12)
                    & (wof_modeluitvoer[var][onderdeel]["time.day"] == 31)
                ]
                wof_modeluitvoer_cum[var][onderdeel]["31okt"] = wof_modeluitvoer[var][
                    onderdeel
                ][
                    (wof_modeluitvoer[var][onderdeel]["time.month"] == 10)
                    & (wof_modeluitvoer[var][onderdeel]["time.day"] == 31)
                ]

        ## LGN kaart in juiste extent zetten
        xmin = wof_modeluitvoer_cum[var][onderdeel]["31dec"].isel(time=0).x.min().values
        xmax = wof_modeluitvoer_cum[var][onderdeel]["31dec"].isel(time=0).x.max().values
        ymin = wof_modeluitvoer_cum[var][onderdeel]["31dec"].isel(time=0).y.min().values
        ymax = wof_modeluitvoer_cum[var][onderdeel]["31dec"].isel(time=0).y.max().values
        lgn_ext = lgn.sel(x=slice(xmin, xmax)).sel(y=slice(ymax, ymin))

        #########################################################
        ## Bij gras ook het maaibare gewas op 31 okt toevoegen ##
        #########################################################

        ## NB. dit doe ik alleen voor stems-leaves en 31 oktober, omdat dat de enige term is die wordt gebruikt voor gras

        ## Maaibare gewas op het land bepalen
        maaibaar_act = (
            wof_modeluitvoer_cum["act"]["stems_leaves"]["31okt"] - stoppelgrootte_gras
        )
        maaibaar_pot = (
            wof_modeluitvoer_cum["pot"]["stems_leaves"]["31okt"] - stoppelgrootte_gras
        )
        ## Als dit kleiner is dan 0, dan toch op 0 zetten
        maaibaar_act = maaibaar_act.where(maaibaar_act > 0, 0)
        maaibaar_pot = maaibaar_pot.where(maaibaar_act > 0, 0)

        ## Yield_pot achterhalen (want dit is geen directe WOFOST uitvoer) (dmv factor tussen cumulatieve pot en act)
        yld_pot = wof_modeluitvoer_cum["yld_act"]["stems_leaves"]["31okt"] / (
            1 - (wof_modeluitvoer_cum["yld_dam"]["stems_leaves"]["31okt"] / 100)
        )

        # for year in range(startjaar,eindjaar + 1):
        # wof_uitvoer_leaves_act_cumsum = (wof_modeluitvoer['act']['stems_leaves'][(wof_modeluitvoer['act']['stems_leaves']['time.year'] == year)] - stoppelgrootte_gras).cumsum()
        # wof_uitvoer_leaves_pot_cumsum = (wof_modeluitvoer['pot']['stems_leaves'][(wof_modeluitvoer['pot']['stems_leaves']['time.year'] == year)] - stoppelgrootte_gras).cumsum()
        # factor = 1 + (wof_uitvoer_leaves_pot_cumsum - wof_uitvoer_leaves_act_cumsum ) / wof_uitvoer_leaves_pot_cumsum

        # yld_pot[yld_pot.time.dt.year == year] = wof_modeluitvoer_cum['yld_act']['stems_leaves']['31okt'][(wof_modeluitvoer_cum['yld_act']['stems_leaves']['31okt']['time.year'] == year)] * factor[(factor.time.dt.month == 10) & (factor.time.dt.day == 31)]

        ## Maaibare onderdeel toevoegen aan yld_act en yld_pot
        wof_modeluitvoer_cum["yld_act"]["stems_leaves"]["31okt"] = (
            wof_modeluitvoer_cum["yld_act"]["stems_leaves"]["31okt"] + maaibaar_act
        )
        yld_pot = yld_pot + maaibaar_pot

        ## Yld_rel opnieuw bepalen (obv act en pot, omdat gras geen IND term heeft)
        ## *100 OMDAT DE
        yld_rel = wof_modeluitvoer_cum["yld_act"]["stems_leaves"]["31okt"] / yld_pot
        # wof_modeluitvoer_cum['yld_rel']['stems_leaves']['31okt'] = wof_modeluitvoer_cum['yld_act']['stems_leaves']['31okt'] / yld_pot
        wof_modeluitvoer_cum["ind"]["stems_leaves"]["31okt"] = yld_pot

        ## Opnieuw yld_dam_tot bepalen (*100 omdat de yld_dam in procenten is)
        ## Maar eerst de oude yld_dam bewaren
        yld_dam_old = wof_modeluitvoer_cum["yld_dam"]["stems_leaves"]["31okt"].copy()
        wof_modeluitvoer_cum["yld_dam"]["stems_leaves"]["31okt"] = (1 - yld_rel) * 100

        ## Factor tussen yld_dam en yld_dam_old bepalen
        factor_yld_dam = (
            wof_modeluitvoer_cum["yld_dam"]["stems_leaves"]["31okt"] / yld_dam_old
        )

        ## En de andere yld_dam's aanpassen, evenredig aan de aanpassing van yld_dam
        for schade in wofost_schadetermen:
            if schade != "dam":  # want die hebben we hierboven al aangepast
                wof_modeluitvoer_cum["yld_" + schade]["stems_leaves"]["31okt"] = (
                    wof_modeluitvoer_cum["yld_" + schade]["stems_leaves"]["31okt"]
                    * factor_yld_dam
                )

        ###############################################################################
        ## Plaats op 31 dec de hoogste opbrengst van dat jaar                    ######
        ## dit brengt geen problemen voor gras, want hierbij kijk je naar 31 okt ######
        ###############################################################################
        for year in range(startjaar, eindjaar + 1):
            for onderdeel in wofost_onderdelen:
                data_act = wof_modeluitvoer["act"][onderdeel][
                    (wof_modeluitvoer["act"][onderdeel]["time.year"] == year)
                ]
                data_ind = wof_modeluitvoer["ind"][onderdeel][
                    (wof_modeluitvoer["ind"][onderdeel]["time.year"] == year)
                ]

                wof_modeluitvoer_cum["act"][onderdeel]["31dec"][
                    (
                        wof_modeluitvoer_cum["act"][onderdeel]["31dec"]["time.year"]
                        == year
                    )
                ] = data_act.max(dim="time")
                wof_modeluitvoer_cum["ind"][onderdeel]["31dec"][
                    (
                        wof_modeluitvoer_cum["ind"][onderdeel]["31dec"]["time.year"]
                        == year
                    )
                ] = data_ind.max(dim="time")

        ###############################
        ## Termen per jaar berekenen ##
        ###############################

        ## Lege dictionaries maken
        wof_ref_opbrengst = dict.fromkeys(wofost_onderdelen)
        wof_schade_kgha = dict.fromkeys(schadetermen_uitgebreid)

        ## referentie opbrengst berekenen
        for onderdeel in wofost_onderdelen:
            wof_ref_opbrengst[onderdeel] = dict.fromkeys(datums)
            for datum in datums:
                # wof_ref_opbrengst[onderdeel][datum] = wof_modeluitvoer_cum['yld_act'][onderdeel][datum] / wof_modeluitvoer_cum['yld_rel'][onderdeel][datum]
                wof_ref_opbrengst[onderdeel][datum] = wof_modeluitvoer_cum["ind"][
                    onderdeel
                ][datum]

        ## Directe schade en natschade berekenen
        wof_modeluitvoer_cum["yld_schade_direct"] = dict.fromkeys(wofost_onderdelen)
        wof_modeluitvoer_cum["yld_schade_nat"] = dict.fromkeys(wofost_onderdelen)
        for onderdeel in wofost_onderdelen:
            wof_modeluitvoer_cum["yld_schade_direct"][onderdeel] = dict.fromkeys(datums)
            wof_modeluitvoer_cum["yld_schade_nat"][onderdeel] = dict.fromkeys(datums)
            for datum in datums:
                wof_modeluitvoer_cum["yld_schade_direct"][onderdeel][datum] = (
                    wof_modeluitvoer_cum["yld_dam_dry"][onderdeel][datum]
                    + wof_modeluitvoer_cum["yld_dam_wet"][onderdeel][datum]
                    + wof_modeluitvoer_cum["yld_dam_salt"][onderdeel][datum]
                )
                wof_modeluitvoer_cum["yld_schade_nat"][onderdeel][datum] = (
                    wof_modeluitvoer_cum["yld_dam_ind"][onderdeel][datum]
                    + wof_modeluitvoer_cum["yld_dam_wet"][onderdeel][datum]
                )

        ## Schadetermen berekenen in kg/ha
        for schade in schadetermen_uitgebreid:
            ## Dictionary maken
            wof_schade_kgha[schade] = dict.fromkeys(wofost_onderdelen)
            for onderdeel in wofost_onderdelen:
                wof_schade_kgha[schade][onderdeel] = dict.fromkeys(datums)
                for datum in datums:
                    ## Schadetermen berekenen in kg/ha
                    wof_schade_kgha[schade][onderdeel][datum] = (
                        (1 / 100)
                        * wof_ref_opbrengst[onderdeel][datum]
                        * wof_modeluitvoer_cum["yld_" + schade][onderdeel][datum]
                    )

        #################################
        ## Sommatie van schadefracties ##
        #################################

        ## Lege dictionaries maken
        wof_ref_opbrengst_som = dict.fromkeys(wofost_onderdelen)
        wof_yld_act_opbrengst_som = dict.fromkeys(wofost_onderdelen)
        wof_act_opbrengst_som = dict.fromkeys(wofost_onderdelen)
        wof_schade_kgha_som = dict.fromkeys(schadetermen_uitgebreid)

        ## Som van referentie opbrengst
        for onderdeel in wofost_onderdelen:
            wof_ref_opbrengst_som[onderdeel] = dict.fromkeys(datums)
            for datum in datums:
                wof_ref_opbrengst_som[onderdeel][datum] = wof_ref_opbrengst[onderdeel][
                    datum
                ].sum("time")

        ## Som van actuele opbrengst
        for onderdeel in wofost_onderdelen:
            wof_yld_act_opbrengst_som[onderdeel] = dict.fromkeys(datums)
            wof_act_opbrengst_som[onderdeel] = dict.fromkeys(datums)
            for datum in datums:
                wof_yld_act_opbrengst_som[onderdeel][datum] = wof_modeluitvoer_cum[
                    "yld_act"
                ][onderdeel][datum].sum("time")
                wof_act_opbrengst_som[onderdeel][datum] = wof_modeluitvoer_cum["act"][
                    onderdeel
                ][datum].sum("time")

        ## Som van schadetermen
        for schade in schadetermen_uitgebreid:
            ## Dictionary maken
            wof_schade_kgha_som[schade] = dict.fromkeys(wofost_onderdelen)
            for onderdeel in wofost_onderdelen:
                wof_schade_kgha_som[schade][onderdeel] = dict.fromkeys(datums)
                for datum in datums:
                    ## Sommatie over de jaren
                    wof_schade_kgha_som[schade][onderdeel][datum] = wof_schade_kgha[
                        schade
                    ][onderdeel][datum].sum("time")

        #################################################
        ## Koppelen aan gewastypen - langjarige sommen ##
        #################################################

        ## Lege dictionaries maken
        ref_opbrengst_per_gewas = dict.fromkeys(["bouwland", "mais", "gras", "totaal"])
        act_opbrengst_per_gewas = dict.fromkeys(["bouwland", "mais", "gras", "totaal"])
        schade_kgha_per_gewas = dict.fromkeys(schadetermen_uitgebreid)

        ## Referentieopbrengst per gewasonderdeel
        ref_opbrengst_per_gewas["bouwland"] = wof_ref_opbrengst_som["storg_organs"][
            "31dec"
        ]
        ref_opbrengst_per_gewas["mais"] = (
            wof_ref_opbrengst_som["storg_organs"]["31dec"]
            + wof_ref_opbrengst_som["stems_leaves"]["31dec"]
        )
        ref_opbrengst_per_gewas["gras"] = wof_ref_opbrengst_som["stems_leaves"]["31okt"]
        ref_opbrengst_per_gewas["totaal"] = (
            ref_opbrengst_per_gewas["bouwland"].where(lgn_ext.isin(LGN_bouwland), 0)
            + ref_opbrengst_per_gewas["mais"].where(lgn_ext.isin(LGN_mais), 0)
            + ref_opbrengst_per_gewas["gras"].where(lgn_ext.isin(LGN_gras), 0)
        )

        ## Actuele opbrengst per gewasonderdeel
        act_opbrengst_per_gewas["bouwland"] = wof_act_opbrengst_som["storg_organs"][
            "31dec"
        ]
        act_opbrengst_per_gewas["mais"] = (
            wof_act_opbrengst_som["storg_organs"]["31dec"]
            + wof_act_opbrengst_som["stems_leaves"]["31dec"]
        )
        act_opbrengst_per_gewas["gras"] = wof_yld_act_opbrengst_som["stems_leaves"][
            "31okt"
        ]
        act_opbrengst_per_gewas["totaal"] = (
            act_opbrengst_per_gewas["bouwland"].where(lgn_ext.isin(LGN_bouwland), 0)
            + act_opbrengst_per_gewas["mais"].where(lgn_ext.isin(LGN_mais), 0)
            + act_opbrengst_per_gewas["gras"].where(lgn_ext.isin(LGN_gras), 0)
        )

        ## Schade per gewasonderdeel
        for schade in schadetermen_uitgebreid:
            schade_kgha_per_gewas[schade] = dict.fromkeys(
                ["bouwland", "mais", "gras", "totaal"]
            )
            schade_kgha_per_gewas[schade]["bouwland"] = wof_schade_kgha_som[schade][
                "storg_organs"
            ]["31dec"]
            schade_kgha_per_gewas[schade]["mais"] = (
                wof_schade_kgha_som[schade]["storg_organs"]["31dec"]
                + wof_schade_kgha_som[schade]["stems_leaves"]["31dec"]
            )
            schade_kgha_per_gewas[schade]["gras"] = wof_schade_kgha_som[schade][
                "stems_leaves"
            ]["31okt"]
            schade_kgha_per_gewas[schade]["totaal"] = (
                schade_kgha_per_gewas[schade]["bouwland"].where(
                    lgn_ext.isin(LGN_bouwland), 0
                )
                + schade_kgha_per_gewas[schade]["mais"].where(lgn_ext.isin(LGN_mais), 0)
                + schade_kgha_per_gewas[schade]["gras"].where(lgn_ext.isin(LGN_gras), 0)
            )

        ########################################
        ## Koppelen aan gewastypen - per jaar ##
        ########################################

        ## Lege dictionaries maken
        ref_opbrengst_jr_per_gewas = dict.fromkeys(
            ["bouwland", "mais", "gras", "totaal"]
        )
        act_yld_opbrengst_jr_per_gewas = dict.fromkeys(
            ["bouwland", "mais", "gras", "totaal"]
        )
        act_opbrengst_jr_per_gewas = dict.fromkeys(
            ["bouwland", "mais", "gras", "totaal"]
        )
        schade_jr_kgha_per_gewas = dict.fromkeys(schadetermen_uitgebreid)

        ## Jaartal toevoegen en datum verwijderen
        ## --> nodig omdat gras 31 oktober gebruikt, rest 31 december. dus die kan je niet samenvoegen
        wof_ref_opbrengst_jr = copy.deepcopy(
            wof_ref_opbrengst
        )  ## deepcopy nodig zodat je niet de originele dict gaat aanpassen
        wof_act_yld_opbrengst_jr = copy.deepcopy(wof_modeluitvoer_cum["yld_act"])
        wof_act_opbrengst_jr = copy.deepcopy(wof_modeluitvoer_cum["act"])
        wof_schade_kgha_jr = copy.deepcopy(wof_schade_kgha)
        for onderdeel in wofost_onderdelen:
            for datum in datums:
                wof_ref_opbrengst_jr[onderdeel][datum]["time"] = wof_ref_opbrengst_jr[
                    onderdeel
                ][datum].time.dt.year
                wof_act_yld_opbrengst_jr[onderdeel][datum]["time"] = (
                    wof_act_yld_opbrengst_jr[onderdeel][datum].time.dt.year
                )
                wof_act_opbrengst_jr[onderdeel][datum]["time"] = wof_act_opbrengst_jr[
                    onderdeel
                ][datum].time.dt.year
                for schade in schadetermen_uitgebreid:
                    wof_schade_kgha_jr[schade][onderdeel][datum]["time"] = (
                        wof_schade_kgha_jr[schade][onderdeel][datum].time.dt.year
                    )

        ## Referentieopbrengst per gewasonderdeel
        ref_opbrengst_jr_per_gewas["bouwland"] = wof_ref_opbrengst_jr["storg_organs"][
            "31dec"
        ]
        ref_opbrengst_jr_per_gewas["mais"] = (
            wof_ref_opbrengst_jr["storg_organs"]["31dec"]
            + wof_ref_opbrengst_jr["stems_leaves"]["31dec"]
        )
        ref_opbrengst_jr_per_gewas["gras"] = wof_ref_opbrengst_jr["stems_leaves"][
            "31okt"
        ]
        ref_opbrengst_jr_per_gewas["totaal"] = (
            ref_opbrengst_jr_per_gewas["bouwland"].where(lgn_ext.isin(LGN_bouwland), 0)
            + ref_opbrengst_jr_per_gewas["mais"].where(lgn_ext.isin(LGN_mais), 0)
            + ref_opbrengst_jr_per_gewas["gras"].where(lgn_ext.isin(LGN_gras), 0)
        )

        ## Actuele opbrengst per gewasonderdeel
        act_opbrengst_jr_per_gewas["bouwland"] = wof_act_opbrengst_jr["storg_organs"][
            "31dec"
        ]
        act_opbrengst_jr_per_gewas["mais"] = (
            wof_act_opbrengst_jr["storg_organs"]["31dec"]
            + wof_act_opbrengst_jr["stems_leaves"]["31dec"]
        )
        act_opbrengst_jr_per_gewas["gras"] = wof_act_yld_opbrengst_jr["stems_leaves"][
            "31okt"
        ]
        act_opbrengst_jr_per_gewas["totaal"] = (
            act_opbrengst_jr_per_gewas["bouwland"].where(lgn_ext.isin(LGN_bouwland), 0)
            + act_opbrengst_jr_per_gewas["mais"].where(lgn_ext.isin(LGN_mais), 0)
            + act_opbrengst_jr_per_gewas["gras"].where(lgn_ext.isin(LGN_gras), 0)
        )

        ## Schade per gewasonderdeel
        for schade in schadetermen_uitgebreid:
            schade_jr_kgha_per_gewas[schade] = dict.fromkeys(
                ["bouwland", "mais", "gras", "totaal"]
            )
            schade_jr_kgha_per_gewas[schade]["bouwland"] = wof_schade_kgha_jr[schade][
                "storg_organs"
            ]["31dec"]
            schade_jr_kgha_per_gewas[schade]["mais"] = (
                wof_schade_kgha_jr[schade]["storg_organs"]["31dec"]
                + wof_schade_kgha_jr[schade]["stems_leaves"]["31dec"]
            )
            schade_jr_kgha_per_gewas[schade]["gras"] = wof_schade_kgha_jr[schade][
                "stems_leaves"
            ]["31okt"]
            schade_jr_kgha_per_gewas[schade]["totaal"] = (
                schade_jr_kgha_per_gewas[schade]["bouwland"].where(
                    lgn_ext.isin(LGN_bouwland), 0
                )
                + schade_jr_kgha_per_gewas[schade]["mais"].where(
                    lgn_ext.isin(LGN_mais), 0
                )
                + schade_jr_kgha_per_gewas[schade]["gras"].where(
                    lgn_ext.isin(LGN_gras), 0
                )
            )

        #######################
        ## Schadepercentages ##
        #######################

        schade_perc = dict.fromkeys(schadetermen_uitgebreid)
        schade_perc_jaar = dict.fromkeys(schadetermen_uitgebreid)

        ## Schadepercentages berekenen
        for schade in schadetermen_uitgebreid:
            ## Langjarig gemiddeld
            schade_perc[schade] = (
                schade_kgha_per_gewas[schade]["totaal"]
                / ref_opbrengst_per_gewas["totaal"]
            )
            ## En per jaar
            schade_perc_jaar[schade] = (
                schade_jr_kgha_per_gewas[schade]["totaal"]
                / ref_opbrengst_jr_per_gewas["totaal"]
            )

        ##########################
        ## Uitvoer wegschrijven ##
        ##########################

        ## Referentieopbrengst (kg/ha)
        imod.idf.write(
            rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/referentieopbrengst_{startjaar}_{eindjaar}_kgha.idf",
            ref_opbrengst_per_gewas["totaal"] / aantal_jaar,
        )
        for j in range(startjaar, eindjaar + 1):
            imod.idf.write(
                rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/referentieopbrengst_{str(j)}_kgha.idf",
                ref_opbrengst_jr_per_gewas["totaal"].sel(time=j),
            )

        ## Schadepercentages
        for schade in schadetermen_uitgebreid:
            imod.idf.write(
                rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/{schade}_{startjaar}_{eindjaar}_perc.idf",
                schade_perc[schade],
            )
            for j in range(startjaar, eindjaar + 1):
                imod.idf.write(
                    rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/{schade}_{str(j)}_perc.idf",
                    schade_perc_jaar[schade].sel(time=j),
                )

        ## Schades (kg/ha)
        for schade in schadetermen_uitgebreid:
            imod.idf.write(
                rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/{schade}_{startjaar}_{eindjaar}_kgha.idf",
                schade_kgha_per_gewas[schade]["totaal"] / aantal_jaar,
            )
            for j in range(startjaar, eindjaar + 1):
                imod.idf.write(
                    rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/{schade}_{str(j)}_kgha.idf",
                    schade_jr_kgha_per_gewas[schade]["totaal"].sel(time=j),
                )

        ## Actuele opbrengst (kg/ha)
        imod.idf.write(
            rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/actuele_opbrengst_{startjaar}_{eindjaar}_kgha.idf",
            act_opbrengst_per_gewas["totaal"] / aantal_jaar,
        )
        for j in range(startjaar, eindjaar + 1):
            imod.idf.write(
                rf"{r}/postprocessing_{startjaar}_{eindjaar}/Landbouwschade/actuele_opbrengst_{str(j)}_kgha.idf",
                act_opbrengst_jr_per_gewas["totaal"].sel(time=j),
            )
