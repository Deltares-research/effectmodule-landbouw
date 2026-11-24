from pathlib import Path

import imod

from input_processing.wofost import (
    DAMAGE_TERMS,
    GRASS_STUBBLE_KG_HA,
    correct_grass_yield,
    load_wofost_data,
    select_annual_outputs,
)


def main() -> None:
    """Prepare WOFOST yield inputs for the Effectmodule Landbouw."""

    startjaar = 2014
    eindjaar = 2015

    run_dir = Path("")

    lgn_path = Path(
        "p:/doorontwikkeling-dpzw-effectmodule/"
        "Landbouw/Data/LGN/LHM_4.3.3/LGN250.IDF"
    )

    output_folder = Path(
        "p:/doorontwikkeling-dpzw-effectmodule/" "Landbouw/Data/WOFOST_LHM_4p3p3/"
    )

    # load land use
    lgn = imod.idf.open(lgn_path)

    wofost_data = load_wofost_data(run_dir, startjaar, eindjaar)
    annual_outputs = select_annual_outputs(wofost_data)
    corrected_annual_outputs = correct_grass_yield(
        annual_outputs, stubble_yield=GRASS_STUBBLE_KG_HA, damage_terms=DAMAGE_TERMS
    )


if __name__ == "__main__":
    main()
