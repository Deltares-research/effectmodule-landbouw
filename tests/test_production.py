import numpy as np
import xarray as xr

from src.processing.prijstool.production import (
    calculate_production_and_costs,
)


def test_calculate_production_and_costs():
    cropdata_sce = xr.Dataset(
        {
            "areaal": ("Year", [2.0, 4.0]),
            "derv_opbrengst": ("Year", [10.0, 20.0]),
            "arbeidskosten": ("Year", [3.0, 5.0]),
            "energiekosten": ("Year", [2.0, 4.0]),
        },
        coords={
            "Year": [2000, 2001],
        },
    )

    cropdata_ref = xr.Dataset(
        {
            "areaal": ("Year", [4.0, 6.0]),
        },
        coords={
            "Year": [2000, 2001],
        },
    )

    standard_yield_sce = xr.DataArray(100.0)
    standard_yield_ref = xr.DataArray(120.0)

    (
        areaal_sce,
        standard_production_ref,
        standard_production_sce,
        yield_loss_sce,
        labour_costs_sce,
        energy_costs_sce,
    ) = calculate_production_and_costs(
        cropdata_sce=cropdata_sce,
        cropdata_ref=cropdata_ref,
        standard_yield_sce=standard_yield_sce,
        standard_yield_ref=standard_yield_ref,
        multpotyield=1.0,
        multredyield=1.0,
    )

    assert areaal_sce.item() == 3.0
    assert standard_production_ref.item() == 600.0
    assert standard_production_sce.item() == 300.0

    np.testing.assert_array_equal(
        yield_loss_sce.values,
        [30.0, 60.0],
    )

    np.testing.assert_array_equal(
        labour_costs_sce.values,
        [9.0, 15.0],
    )

    np.testing.assert_array_equal(
        energy_costs_sce.values,
        [6.0, 12.0],
    )
