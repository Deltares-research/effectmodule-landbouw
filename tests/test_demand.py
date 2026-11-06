import numpy as np
import xarray as xr

from src.processing.prijstool.demand import calibrate_linear_demand


def test_calibrate_linear_demand_local_market():
    standard_production_local = xr.DataArray([100.0])
    standard_production_world = xr.DataArray([500.0])
    base_price = xr.DataArray([2.0])

    elasticity = xr.Dataset(
        {
            "elasticity": xr.DataArray([-0.5]),
        }
    )

    local_market_price_effect = xr.DataArray([True])
    world_market_price_effect = xr.DataArray([False])

    demand_slope, demand_intercept = calibrate_linear_demand(
        standard_production=standard_production_local,
        standard_production_world=standard_production_world,
        base_price=base_price,
        elasticity=elasticity,
        local_market_price_effect=local_market_price_effect,
        world_market_price_effect=world_market_price_effect,
    )

    np.testing.assert_allclose(
        demand_slope.values,
        [-25.0],
    )

    np.testing.assert_allclose(
        demand_intercept.values,
        [150.0],
    )

    reconstructed_production = demand_slope * base_price + demand_intercept

    np.testing.assert_allclose(
        reconstructed_production.values,
        standard_production_local.values,
    )


def test_calibrate_linear_demand_world_market():
    standard_production_local = xr.DataArray([100.0])
    standard_production_world = xr.DataArray([500.0])
    base_price = xr.DataArray([2.0])

    elasticity = xr.Dataset(
        {
            "elasticity": xr.DataArray([-0.5]),
        }
    )

    local_market_price_effect = xr.DataArray([False])
    world_market_price_effect = xr.DataArray([True])

    demand_slope, demand_intercept = calibrate_linear_demand(
        standard_production=standard_production_world,
        standard_production_world=standard_production_world,
        base_price=base_price,
        elasticity=elasticity,
        local_market_price_effect=local_market_price_effect,
        world_market_price_effect=world_market_price_effect,
    )

    np.testing.assert_allclose(
        demand_slope.values,
        [-125.0],
    )

    np.testing.assert_allclose(
        demand_intercept.values,
        [750.0],
    )

    reconstructed_production = demand_slope * base_price + demand_intercept

    np.testing.assert_allclose(
        reconstructed_production.values,
        standard_production_world.values,
    )
