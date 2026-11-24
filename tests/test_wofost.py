import numpy as np
import pandas as pd
import xarray as xr

from src.input_processing.wofost import (
    DAMAGE_TERMS,
    WOFOST_COMPONENTS,
    WOFOST_VARIABLES,
    correct_grass_yield,
    select_annual_outputs,
)


def create_grass_test_data(
    act=1000.0,
    pot=1200.0,
    yld_act=8000.0,
    yld_dam=20.0,
):
    data = {
        "act": {
            "stems_leaves": {
                "grass": xr.DataArray([act]),
            }
        },
        "pot": {
            "stems_leaves": {
                "grass": xr.DataArray([pot]),
            }
        },
        "ind": {
            "stems_leaves": {
                "grass": xr.DataArray([0.0]),
            }
        },
        "yld_act": {
            "stems_leaves": {
                "grass": xr.DataArray([yld_act]),
            }
        },
        "yld_dam": {
            "stems_leaves": {
                "grass": xr.DataArray([yld_dam]),
            }
        },
        "yld_dam_dry": {
            "stems_leaves": {
                "grass": xr.DataArray([10.0]),
            }
        },
        "yld_dam_ind": {
            "stems_leaves": {
                "grass": xr.DataArray([5.0]),
            }
        },
        "yld_dam_salt": {
            "stems_leaves": {
                "grass": xr.DataArray([3.0]),
            }
        },
        "yld_dam_wet": {
            "stems_leaves": {
                "grass": xr.DataArray([2.0]),
            }
        },
    }

    return data


def test_select_annual_outputs():
    time = pd.to_datetime(
        [
            "2014-10-31",
            "2014-12-31",
            "2015-10-31",
            "2015-12-31",
        ]
    )

    data = xr.DataArray(
        [10.0, 20.0, 30.0, 40.0],
        dims="time",
        coords={"time": time},
    )

    wofost_data = {}

    for variable in WOFOST_VARIABLES:
        wofost_data[variable] = {}

        for component in WOFOST_COMPONENTS:
            wofost_data[variable][component] = data.copy()

    result = select_annual_outputs(wofost_data)

    grass = result["act"]["stems_leaves"]["grass"]

    other_crops = result["act"]["stems_leaves"]["other_crops"]

    np.testing.assert_array_equal(
        grass.values,
        [10.0, 30.0],
    )

    np.testing.assert_array_equal(
        other_crops.values,
        [20.0, 40.0],
    )

    np.testing.assert_array_equal(
        grass.time.dt.month.values,
        [10, 10],
    )

    np.testing.assert_array_equal(
        grass.time.dt.day.values,
        [31, 31],
    )

    np.testing.assert_array_equal(
        other_crops.time.dt.month.values,
        [12, 12],
    )

    np.testing.assert_array_equal(
        other_crops.time.dt.day.values,
        [31, 31],
    )


def test_correct_grass_yield_positive_harvestable_biomass():
    annual_outputs = create_grass_test_data()

    result = correct_grass_yield(
        annual_outputs=annual_outputs,
        stubble_yield=700,
        damage_terms=DAMAGE_TERMS,
    )

    corrected_actual = result["yld_act"]["stems_leaves"]["grass"]

    corrected_potential = result["ind"]["stems_leaves"]["grass"]

    np.testing.assert_allclose(
        corrected_actual.values,
        [8300.0],
    )

    np.testing.assert_allclose(
        corrected_potential.values,
        [10500.0],
    )

    corrected_damage = result["yld_dam"]["stems_leaves"]["grass"]

    expected_damage = (1 - 8300.0 / 10500.0) * 100

    np.testing.assert_allclose(
        corrected_damage.values,
        [expected_damage],
    )


def test_correct_grass_yield_below_stubble_threshold():
    annual_outputs = create_grass_test_data(
        act=600.0,
        pot=650.0,
        yld_act=8000.0,
        yld_dam=20.0,
    )

    result = correct_grass_yield(
        annual_outputs=annual_outputs,
        stubble_yield=700,
        damage_terms=DAMAGE_TERMS,
    )

    corrected_actual = result["yld_act"]["stems_leaves"]["grass"]

    corrected_potential = result["ind"]["stems_leaves"]["grass"]

    np.testing.assert_allclose(
        corrected_actual.values,
        [8000.0],
    )

    np.testing.assert_allclose(
        corrected_potential.values,
        [10000.0],
    )


def test_correct_grass_yield_scales_damage_components():
    annual_outputs = create_grass_test_data()

    result = correct_grass_yield(
        annual_outputs=annual_outputs,
        stubble_yield=700,
        damage_terms=DAMAGE_TERMS,
    )

    new_total_damage = result["yld_dam"]["stems_leaves"]["grass"].item()

    adjustment_factor = new_total_damage / 20.0

    expected_dry_damage = 10.0 * adjustment_factor

    corrected_dry_damage = result["yld_dam_dry"]["stems_leaves"]["grass"]

    np.testing.assert_allclose(
        corrected_dry_damage.values,
        [expected_dry_damage],
    )
