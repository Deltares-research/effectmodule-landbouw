from datetime import datetime
from pathlib import Path

import imod

from src.input_processing.fao import (
    FAO_LGN_CLASSES,
    calculate_fao_yields,
    create_ky_grid,
    create_potential_yield_grid,
    load_transpiration,
    prepare_reference_tpot,
)


def main() -> None:
    start_time = datetime.now()

    startjaar = 2014
    eindjaar = 2015

    path_lgn = Path(
        "p:/doorontwikkeling-dpzw-effectmodule/"
        "Landbouw/Data/LGN/LHM_4.3.3/LGN250.IDF"
    )

    path_data = (
        r"p:\archivedprojects\11210362-dpzw-basisprognoses" r"\bp-zw-24\postprocessed"
    )

    path_output = Path(
        "p:/doorontwikkeling-dpzw-effectmodule/" "Landbouw/Data/FAO schades/"
    )

    folder = Path(path_data) / "REF2028" / "REF2028a"

    tpot_files = [
        folder / f"Tpot_{year}0101000000.asc" for year in range(startjaar, eindjaar + 1)
    ]

    tact_files = [
        folder / f"Tact_{year}0101000000.asc" for year in range(startjaar, eindjaar + 1)
    ]

    # Load land use
    lgn = imod.idf.open(path_lgn)

    # Create FAO crop inputs
    ky = create_ky_grid(lgn)

    yldpot_langjarig = create_potential_yield_grid(lgn)

    tpot_langjarig = prepare_reference_tpot(
        tpot_files,
        lgn,
        FAO_LGN_CLASSES,
    )

    valid_reference_tpot = tpot_langjarig.notnull() & (tpot_langjarig != 0)

    # Load transpiration
    msw_tpot = load_transpiration(tpot_files).where(valid_reference_tpot)

    msw_tact = load_transpiration(tact_files).where(valid_reference_tpot)

    # Calculate yields
    opbrengst = calculate_fao_yields(
        startjaar,
        eindjaar,
        msw_tact,
        msw_tpot,
        tpot_langjarig,
        yldpot_langjarig,
        ky,
    )

    # Write outputs
    imod.idf.save(
        path_output / "actuele_opbrengst.idf",
        opbrengst["actueel"],
        pattern="{name}_{time.year}.idf",
    )

    imod.idf.save(
        path_output / "potentiele_opbrengst.idf",
        opbrengst["potentieel"],
        pattern="{name}_{time.year}.idf",
    )

    imod.idf.save(
        path_output / "relatieve_opbrengst.idf",
        opbrengst["relatief"],
        pattern="{name}_{time.year}.idf",
    )

    print(f"Duration: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
