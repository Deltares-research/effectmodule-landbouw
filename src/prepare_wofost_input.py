from pathlib import Path

import imod
import xarray as xr

from input_processing.wofost import (
    DAMAGE_TERMS,
    EXTENDED_DAMAGE_TERMS,
    GRASS_STUBBLE_KG_HA,
    add_direct_and_wet_damage,
    calculate_all_damage_kg_ha,
    calculate_damage_fractions,
    combine_outputs_by_crop,
    correct_grass_yield,
    load_wofost_data,
    prepare_annual_outputs_by_year,
    select_annual_outputs,
    set_annual_maximum_yields,
    sum_outputs_over_time,
)


def write_annual_idfs(
    data: xr.DataArray,
    output_folder: Path,
    name: str,
    start_year: int,
    end_year: int,
    unit: str,
) -> None:
    """Write one IDF file per year."""
    for year in range(start_year, end_year + 1):
        imod.idf.write(
            output_folder / f"{name}_{year}_{unit}.idf",
            data.sel(time=year),
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

    # Align LGN raster with WOFOST output grid
    reference_grid = annual_outputs["act"]["stems_leaves"]["other_crops"].isel(time=0)

    xmin = reference_grid.x.min().item()
    xmax = reference_grid.x.max().item()
    ymin = reference_grid.y.min().item()
    ymax = reference_grid.y.max().item()

    lgn_ext = lgn.sel(x=slice(xmin, xmax)).sel(y=slice(ymax, ymin))

    assert lgn_ext.x.equals(reference_grid.x)
    assert lgn_ext.y.equals(reference_grid.y)

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

    # long-term sums
    summed_outputs = sum_outputs_over_time(
        annual_outputs=annual_outputs,
        damage_kg_ha=damage_kg_ha,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )

    combined_summed_outputs = combine_outputs_by_crop(
        outputs=summed_outputs,
        lgn=lgn_ext,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )

    yearly_outputs = prepare_annual_outputs_by_year(
        annual_outputs=annual_outputs,
        damage_kg_ha=damage_kg_ha,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )

    combined_yearly_outputs = combine_outputs_by_crop(
        outputs=yearly_outputs,
        lgn=lgn_ext,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )

    damage_fractions = calculate_damage_fractions(
        combined_outputs=combined_summed_outputs,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )

    damage_fractions_yearly = calculate_damage_fractions(
        combined_outputs=combined_yearly_outputs,
        damage_terms=EXTENDED_DAMAGE_TERMS,
    )

    number_of_years = eindjaar - startjaar + 1

    imod.idf.write(
        output_folder / f"referentieopbrengst_{startjaar}_{eindjaar}_kgha.idf",
        (combined_summed_outputs["reference_yield"]["total"] / number_of_years),
    )

    write_annual_idfs(
        data=combined_yearly_outputs["reference_yield"]["total"],
        output_folder=output_folder,
        name="referentieopbrengst",
        start_year=startjaar,
        end_year=eindjaar,
        unit="kgha",
    )

    imod.idf.write(
        output_folder / f"actuele_opbrengst_{startjaar}_{eindjaar}_kgha.idf",
        (combined_summed_outputs["actual_yield"]["total"] / number_of_years),
    )

    write_annual_idfs(
        data=combined_yearly_outputs["actual_yield"]["total"],
        output_folder=output_folder,
        name="actuele_opbrengst",
        start_year=startjaar,
        end_year=eindjaar,
        unit="kgha",
    )

    for damage_term in EXTENDED_DAMAGE_TERMS:
        imod.idf.write(
            output_folder / (f"{damage_term}_" f"{startjaar}_{eindjaar}_kgha.idf"),
            (
                combined_summed_outputs["damage_kg_ha"][damage_term]["total"]
                / number_of_years
            ),
        )

        write_annual_idfs(
            data=combined_yearly_outputs["damage_kg_ha"][damage_term]["total"],
            output_folder=output_folder,
            name=damage_term,
            start_year=startjaar,
            end_year=eindjaar,
            unit="kgha",
        )

    for damage_term in EXTENDED_DAMAGE_TERMS:
        imod.idf.write(
            output_folder / (f"{damage_term}_" f"{startjaar}_{eindjaar}_perc.idf"),
            damage_fractions[damage_term],
        )

        write_annual_idfs(
            data=damage_fractions_yearly[damage_term],
            output_folder=output_folder,
            name=damage_term,
            start_year=startjaar,
            end_year=eindjaar,
            unit="perc",
        )


if __name__ == "__main__":
    main()
