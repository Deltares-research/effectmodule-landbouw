import numpy as np
import pandas as pd
import xarray as xr

from src.input_processing.wofost import (
    DAMAGE_TERMS,
    WOFOST_COMPONENTS,
    WOFOST_VARIABLES,
    calculate_all_damage_kg_ha,
    calculate_damage_fractions,
    calculate_damage_kg_ha,
    combine_outputs_by_crop,
    correct_grass_yield,
    prepare_annual_outputs_by_year,
    select_annual_outputs,
    set_annual_maximum_yields,
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


def create_wofost_test_data(time, default_value=1.0):
    data = xr.DataArray(
        np.full(len(time), default_value),
        dims="time",
        coords={"time": time},
    )

    wofost_data = {}

    for variable in WOFOST_VARIABLES:
        wofost_data[variable] = {}

        for component in WOFOST_COMPONENTS:
            wofost_data[variable][component] = data.copy()

    return wofost_data


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


def test_set_annual_maximum_yields():
    time = pd.to_datetime(
        [
            "2014-06-01",
            "2014-10-31",
            "2014-12-31",
            "2015-06-01",
            "2015-10-31",
            "2015-12-31",
        ]
    )

    act_data = xr.DataArray(
        [90.0, 100.0, 80.0, 200.0, 180.0, 150.0],
        dims="time",
        coords={"time": time},
    )

    ind_data = xr.DataArray(
        [120.0, 110.0, 100.0, 220.0, 210.0, 190.0],
        dims="time",
        coords={"time": time},
    )

    # Create the complete WOFOST data structure.
    wofost_data = create_wofost_test_data(time)

    # Replace act and ind with values specifically chosen
    # for testing the annual maximum calculation.
    for component in WOFOST_COMPONENTS:
        wofost_data["act"][component] = act_data.copy()
        wofost_data["ind"][component] = ind_data.copy()

    annual_outputs = select_annual_outputs(wofost_data)

    grass_before = annual_outputs["act"]["stems_leaves"]["grass"].copy()

    result = set_annual_maximum_yields(
        wofost_data=wofost_data,
        annual_outputs=annual_outputs,
        start_year=2014,
        end_year=2015,
    )

    grass_after = result["act"]["stems_leaves"]["grass"]

    xr.testing.assert_equal(
        grass_after,
        grass_before,
    )

    actual_other_crops = result["act"]["stems_leaves"]["other_crops"]

    reference_other_crops = result["ind"]["stems_leaves"]["other_crops"]

    np.testing.assert_array_equal(
        actual_other_crops.values,
        [100.0, 200.0],
    )

    np.testing.assert_array_equal(
        reference_other_crops.values,
        [120.0, 220.0],
    )


def test_calculate_damage_kg_ha():
    reference_yield = xr.DataArray([10000.0, 8000.0, 5000.0])

    damage_percentage = xr.DataArray([20.0, 10.0, 0.0])

    result = calculate_damage_kg_ha(
        reference_yield=reference_yield,
        damage_percentage=damage_percentage,
    )

    expected = np.array([2000.0, 800.0, 0.0])

    np.testing.assert_array_equal(
        result.values,
        expected,
    )


def test_calculate_all_damage_kg_ha():
    annual_outputs = {
        "ind": {
            "stems_leaves": {
                "grass": xr.DataArray([10000.0]),
                "other_crops": xr.DataArray([8000.0]),
            },
            "storg_organs": {
                "grass": xr.DataArray([5000.0]),
                "other_crops": xr.DataArray([6000.0]),
            },
        },
        "yld_dam": {
            "stems_leaves": {
                "grass": xr.DataArray([20.0]),
                "other_crops": xr.DataArray([10.0]),
            },
            "storg_organs": {
                "grass": xr.DataArray([5.0]),
                "other_crops": xr.DataArray([25.0]),
            },
        },
    }

    result = calculate_all_damage_kg_ha(
        annual_outputs=annual_outputs,
        damage_terms=["dam"],
    )

    assert result["dam"]["stems_leaves"]["grass"].item() == 2000.0
    assert result["dam"]["stems_leaves"]["other_crops"].item() == 800.0
    assert result["dam"]["storg_organs"]["grass"].item() == 250.0
    assert result["dam"]["storg_organs"]["other_crops"].item() == 1500.0


def test_combine_outputs_by_crop():
    x = [125.0, 375.0, 625.0, 875.0]

    # LGN:
    # 1 = grass
    # 2 = maize
    # 3 = arable
    # 6 = FAO crop, should not receive WOFOST output
    lgn = xr.DataArray(
        [1.0, 2.0, 3.0, 6.0],
        dims="x",
        coords={"x": x},
    )

    summed_outputs = {
        "reference_yield": {
            "storg_organs": {
                "other_crops": xr.DataArray(
                    [10.0, 10.0, 10.0, 10.0],
                    dims="x",
                    coords={"x": x},
                ),
            },
            "stems_leaves": {
                "other_crops": xr.DataArray(
                    [1.0, 1.0, 1.0, 1.0],
                    dims="x",
                    coords={"x": x},
                ),
                "grass": xr.DataArray(
                    [100.0, 100.0, 100.0, 100.0],
                    dims="x",
                    coords={"x": x},
                ),
            },
        },
        "actual_biomass": {
            "storg_organs": {
                "other_crops": xr.DataArray(
                    [8.0, 8.0, 8.0, 8.0],
                    dims="x",
                    coords={"x": x},
                ),
            },
            "stems_leaves": {
                "other_crops": xr.DataArray(
                    [2.0, 2.0, 2.0, 2.0],
                    dims="x",
                    coords={"x": x},
                ),
            },
        },
        "actual_yield": {
            "stems_leaves": {
                "grass": xr.DataArray(
                    [90.0, 90.0, 90.0, 90.0],
                    dims="x",
                    coords={"x": x},
                ),
            },
        },
        "damage_kg_ha": {
            "dam": {
                "storg_organs": {
                    "other_crops": xr.DataArray(
                        [4.0, 4.0, 4.0, 4.0],
                        dims="x",
                        coords={"x": x},
                    ),
                },
                "stems_leaves": {
                    "other_crops": xr.DataArray(
                        [1.0, 1.0, 1.0, 1.0],
                        dims="x",
                        coords={"x": x},
                    ),
                    "grass": xr.DataArray(
                        [3.0, 3.0, 3.0, 3.0],
                        dims="x",
                        coords={"x": x},
                    ),
                },
            },
        },
    }

    result = combine_outputs_by_crop(
        summed_outputs=summed_outputs,
        lgn=lgn,
        damage_terms=["dam"],
    )

    # Reference yield:
    # grass  -> stems_leaves = 100
    # maize  -> storg_organs + stems_leaves = 10 + 1 = 11
    # arable -> storg_organs = 10
    # LGN 6  -> no WOFOST output = 0
    np.testing.assert_array_equal(
        result["reference_yield"]["total"].values,
        [100.0, 11.0, 10.0, 0.0],
    )

    # Actual yield:
    # grass  -> corrected yld_act = 90
    # maize  -> act organs + leaves = 8 + 2 = 10
    # arable -> act organs = 8
    # LGN 6  -> no WOFOST output = 0
    np.testing.assert_array_equal(
        result["actual_yield"]["total"].values,
        [90.0, 10.0, 8.0, 0.0],
    )

    # Damage:
    # grass  -> stems_leaves = 3
    # maize  -> organs + leaves = 4 + 1 = 5
    # arable -> organs = 4
    # LGN 6  -> no WOFOST output = 0
    np.testing.assert_array_equal(
        result["damage_kg_ha"]["dam"]["total"].values,
        [3.0, 5.0, 4.0, 0.0],
    )


def test_prepare_and_combine_yearly_outputs():
    x = [125.0, 375.0, 625.0, 875.0]

    grass_time = pd.to_datetime(
        [
            "2014-10-31",
            "2015-10-31",
        ]
    )

    other_crops_time = pd.to_datetime(
        [
            "2014-12-31",
            "2015-12-31",
        ]
    )

    lgn = xr.DataArray(
        [1.0, 2.0, 3.0, 6.0],
        dims="x",
        coords={"x": x},
    )

    def create_data(values, time):
        return xr.DataArray(
            values,
            dims=("time", "x"),
            coords={
                "time": time,
                "x": x,
            },
        )

    def create_zeros(time):
        return create_data(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            time,
        )

    annual_outputs = {
        "ind": {
            "storg_organs": {
                "grass": create_zeros(grass_time),
                "other_crops": create_data(
                    [
                        [10.0, 10.0, 10.0, 10.0],
                        [20.0, 20.0, 20.0, 20.0],
                    ],
                    other_crops_time,
                ),
            },
            "stems_leaves": {
                "other_crops": create_data(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                    ],
                    other_crops_time,
                ),
                "grass": create_data(
                    [
                        [100.0, 100.0, 100.0, 100.0],
                        [200.0, 200.0, 200.0, 200.0],
                    ],
                    grass_time,
                ),
            },
        },
        "act": {
            "storg_organs": {
                "grass": create_zeros(grass_time),
                "other_crops": create_data(
                    [
                        [8.0, 8.0, 8.0, 8.0],
                        [16.0, 16.0, 16.0, 16.0],
                    ],
                    other_crops_time,
                ),
            },
            "stems_leaves": {
                "grass": create_zeros(grass_time),
                "other_crops": create_data(
                    [
                        [2.0, 2.0, 2.0, 2.0],
                        [4.0, 4.0, 4.0, 4.0],
                    ],
                    other_crops_time,
                ),
            },
        },
        "yld_act": {
            "storg_organs": {
                "grass": create_zeros(grass_time),
                "other_crops": create_zeros(other_crops_time),
            },
            "stems_leaves": {
                "grass": create_data(
                    [
                        [90.0, 90.0, 90.0, 90.0],
                        [180.0, 180.0, 180.0, 180.0],
                    ],
                    grass_time,
                ),
                "other_crops": create_zeros(other_crops_time),
            },
        },
    }

    damage_kg_ha = {
        "dam": {
            "storg_organs": {
                "grass": create_zeros(grass_time),
                "other_crops": create_data(
                    [
                        [4.0, 4.0, 4.0, 4.0],
                        [8.0, 8.0, 8.0, 8.0],
                    ],
                    other_crops_time,
                ),
            },
            "stems_leaves": {
                "other_crops": create_data(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                    ],
                    other_crops_time,
                ),
                "grass": create_data(
                    [
                        [3.0, 3.0, 3.0, 3.0],
                        [6.0, 6.0, 6.0, 6.0],
                    ],
                    grass_time,
                ),
            },
        },
    }

    yearly_outputs = prepare_annual_outputs_by_year(
        annual_outputs=annual_outputs,
        damage_kg_ha=damage_kg_ha,
        damage_terms=["dam"],
    )

    result = combine_outputs_by_crop(
        outputs=yearly_outputs,
        lgn=lgn,
        damage_terms=["dam"],
    )

    # October and December should now both use integer years.
    np.testing.assert_array_equal(
        result["reference_yield"]["total"].time.values,
        [2014, 2015],
    )

    # Reference yield:
    # grass  -> stems_leaves
    # maize  -> storg_organs + stems_leaves
    # arable -> storg_organs
    # LGN 6  -> FAO, therefore zero
    expected_reference = np.array(
        [
            [100.0, 11.0, 10.0, 0.0],
            [200.0, 22.0, 20.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(
        result["reference_yield"]["total"].values,
        expected_reference,
    )

    # Actual yield:
    # grass uses yld_act
    # maize/arable use act
    expected_actual = np.array(
        [
            [90.0, 10.0, 8.0, 0.0],
            [180.0, 20.0, 16.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(
        result["actual_yield"]["total"].values,
        expected_actual,
    )

    expected_damage = np.array(
        [
            [3.0, 5.0, 4.0, 0.0],
            [6.0, 10.0, 8.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(
        result["damage_kg_ha"]["dam"]["total"].values,
        expected_damage,
    )


def test_calculate_damage_fractions():
    combined_outputs = {
        "reference_yield": {
            "total": xr.DataArray([10000.0, 8000.0]),
        },
        "damage_kg_ha": {
            "dam": {
                "total": xr.DataArray([2000.0, 800.0]),
            },
            "dam_dry": {
                "total": xr.DataArray([1000.0, 400.0]),
            },
        },
    }

    result = calculate_damage_fractions(
        combined_outputs=combined_outputs,
        damage_terms=[
            "dam",
            "dam_dry",
        ],
    )

    np.testing.assert_allclose(
        result["dam"].values,
        [0.20, 0.10],
    )

    np.testing.assert_allclose(
        result["dam_dry"].values,
        [0.10, 0.05],
    )
