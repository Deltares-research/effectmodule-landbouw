from pathlib import Path

import imod

from input_processing.wofost import (
    DAMAGE_TERMS,
    EXTENDED_DAMAGE_TERMS,
    GRASS_STUBBLE_KG_HA,
    add_direct_and_wet_damage,
    calculate_all_damage_kg_ha,
    correct_grass_yield,
    load_wofost_data,
    select_annual_outputs,
    set_annual_maximum_yields,
    sum_outputs_over_time,
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
    annual_outputs = correct_grass_yield(
        annual_outputs, stubble_yield=GRASS_STUBBLE_KG_HA, damage_terms=DAMAGE_TERMS
    )
    annual_outputs = set_annual_maximum_yields(
        wofost_data, annual_outputs, startjaar, eindjaar
    )
    annual_outputs = add_direct_and_wet_damage(annual_outputs)

    damage_kg_ha = calculate_all_damage_kg_ha(
        annual_outputs, damage_terms=EXTENDED_DAMAGE_TERMS
    )

    summed_outputs = sum_outputs_over_time(
        annual_outputs=annual_outputs,
        damage_kg_ha=damage_kg_ha,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )


if __name__ == "__main__":
    main()
