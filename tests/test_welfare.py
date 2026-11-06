import numpy as np
import xarray as xr

from src.processing.prijstool.welfare import (
    calculate_variable_irrigation_costs,
    calculate_welfare_changes,
)


def test_calculate_variable_irrigation_costs():
    arbeidskosten_sce = xr.DataArray(
        [1.0, 2.0, 4.0],
        dims="Year",
        coords={
            "Year": [2000, 2001, 2002],
        },
    )

    energiekosten_sce = xr.DataArray(
        [1.0, 2.0, 6.0],
        dims="Year",
        coords={
            "Year": [2000, 2001, 2002],
        },
    )

    result = calculate_variable_irrigation_costs(
        arbeidskosten_sce=arbeidskosten_sce,
        energiekosten_sce=energiekosten_sce,
        perc_prijspeil_beregening=10.0,
    )

    expected = np.array([0.0, 0.0, 13.2])

    np.testing.assert_allclose(
        result.values,
        expected,
    )


def test_variable_irrigation_costs_median_boundary():
    arbeidskosten_sce = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="Year",
        coords={"Year": [2000, 2001, 2002]},
    )

    energiekosten_sce = xr.DataArray(
        [0.0, 0.0, 0.0],
        dims="Year",
        coords={"Year": [2000, 2001, 2002]},
    )

    result = calculate_variable_irrigation_costs(
        arbeidskosten_sce=arbeidskosten_sce,
        energiekosten_sce=energiekosten_sce,
        perc_prijspeil_beregening=0.0,
    )

    expected = np.array([0.0, 0.0, 2.0])

    np.testing.assert_array_equal(
        result.values,
        expected,
    )


def test_variable_irrigation_costs_even_years_median():
    arbeidskosten_sce = xr.DataArray(
        [1.0, 2.0, 4.0, 8.0],
        dims="Year",
        coords={"Year": [2000, 2001, 2002, 2003]},
    )

    energiekosten_sce = xr.DataArray(
        [0.0, 0.0, 0.0, 0.0],
        dims="Year",
        coords={"Year": [2000, 2001, 2002, 2003]},
    )

    result = calculate_variable_irrigation_costs(
        arbeidskosten_sce=arbeidskosten_sce,
        energiekosten_sce=energiekosten_sce,
        perc_prijspeil_beregening=0.0,
    )

    expected = np.array([0.0, 0.0, 2.0, 10.0])

    np.testing.assert_array_equal(
        result.values,
        expected,
    )


def test_calculate_welfare_changes_with_price_effect():
    variable_irrigation_costs_def = xr.DataArray([10.0])

    new_yield_local = xr.DataArray([80.0])
    new_price = xr.DataArray([3.0])

    standard_opbrengst_kg_sce = xr.DataArray([100.0])
    base_price = xr.DataArray([2.0])

    droogtederving_kg_sce = xr.DataArray([20.0])
    change_consumer_surplus = xr.DataArray([15.0])

    p_elas = xr.Dataset(
        {
            "prijseffect": xr.DataArray([1]),
        }
    )

    (
        change_producer_surplus,
        change_total_welfare,
    ) = calculate_welfare_changes(
        variable_irrigation_costs_def=variable_irrigation_costs_def,
        new_yield_local=new_yield_local,
        new_price=new_price,
        standard_opbrengst_kg_sce=standard_opbrengst_kg_sce,
        base_price=base_price,
        droogtederving_kg_sce=droogtederving_kg_sce,
        change_consumer_surplus=change_consumer_surplus,
        p_elas=p_elas,
    )

    np.testing.assert_array_equal(
        change_producer_surplus.values,
        [30.0],
    )

    np.testing.assert_array_equal(
        change_total_welfare.values,
        [45.0],
    )


def test_calculate_welfare_changes_without_price_effect():
    variable_irrigation_costs_def = xr.DataArray([10.0])

    new_yield_local = xr.DataArray([80.0])
    new_price = xr.DataArray([3.0])

    standard_opbrengst_kg_sce = xr.DataArray([100.0])
    base_price = xr.DataArray([2.0])

    droogtederving_kg_sce = xr.DataArray([20.0])
    change_consumer_surplus = xr.DataArray([0.0])

    p_elas = xr.Dataset(
        {
            "prijseffect": xr.DataArray([0]),
        }
    )

    (
        change_producer_surplus,
        change_total_welfare,
    ) = calculate_welfare_changes(
        variable_irrigation_costs_def=variable_irrigation_costs_def,
        new_yield_local=new_yield_local,
        new_price=new_price,
        standard_opbrengst_kg_sce=standard_opbrengst_kg_sce,
        base_price=base_price,
        droogtederving_kg_sce=droogtederving_kg_sce,
        change_consumer_surplus=change_consumer_surplus,
        p_elas=p_elas,
    )

    np.testing.assert_array_equal(
        change_producer_surplus.values,
        [-50.0],
    )

    np.testing.assert_array_equal(
        change_total_welfare.values,
        [-50.0],
    )
