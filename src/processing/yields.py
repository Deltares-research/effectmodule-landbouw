import xarray as xr


def calculate_median_yields(
    potential: xr.DataArray,
    actual: xr.DataArray,
):
    median_actual = actual.median(dim="time")

    yield_loss = potential - actual
    median_loss = yield_loss.median(dim="time")

    return median_actual, median_loss
