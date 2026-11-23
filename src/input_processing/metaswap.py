from pathlib import Path

import imod
import xarray as xr


def aggregate_decade_transpiration(files: list[Path]) -> xr.DataArray:
    """ "
    Aggregate transpiration from mm/decade to mm/year.
    """

    data = xr.concat(
        [imod.idf.open(path) for path in files],
        dim="time",
    )

    # aggregate per year
    annual_data = data.resample(time="YS").sum(dim="time", skipna=True, min_count=1)

    return annual_data
