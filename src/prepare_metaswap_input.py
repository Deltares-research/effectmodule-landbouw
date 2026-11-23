from pathlib import Path

import xarray as xr

from input_processing.metaswap import aggregate_decade_transpiration


def write_annual_transpiration(
    data: xr.DataArray,
    output_folder: Path,
    variable: str,
    start_year: int,
    end_year: int,
) -> None:
    """Write annual transpiration data to ASCII rasters."""
    for year in range(start_year, end_year + 1):
        output_file = output_folder / f"{variable}_{year}0101000000.asc"

        data.sel(time=f"{year}-01-01").rio.to_raster(
            output_file,
            driver="AAIGrid",
            nodata=-9999,
        )


startjaar = 2014
eindjaar = 2015

folder_data = Path("p:/doorontwikkeling-dpzw-effectmodule/Landbouw/Data/Metaswap/idfs/")
folder_output = Path(
    "p:/doorontwikkeling-dpzw-effectmodule/Landbouw/Data/Metaswap/asciis/"
)

tpot_files = []

tact_files = []

for year in range(startjaar, eindjaar + 1):
    tpot_files.extend(sorted(folder_data.glob(f"msw_Tpot_{year}????_l1.idf")))

    tact_files.extend(sorted(folder_data.glob(f"msw_Tact_{year}????_l1.idf")))

tpot_annual = aggregate_decade_transpiration(tpot_files)
tact_annual = aggregate_decade_transpiration(tact_files)

write_annual_transpiration(
    data=tpot_annual,
    output_folder=folder_output,
    variable="Tpot",
    start_year=startjaar,
    end_year=eindjaar,
)

write_annual_transpiration(
    data=tact_annual,
    output_folder=folder_output,
    variable="Tact",
    start_year=startjaar,
    end_year=eindjaar,
)
