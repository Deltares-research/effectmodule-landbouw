"""
Created on Thu Jun 27 21:16:50 2024

@author: america
"""

"""
Janneke Pouwels, 23-11-2021
Project: DPZW - FAO analyses

Doel script: FAO berekening maken

Opmerkingen
    - Corrigeren voor het metaswap areaal is niet nodig, omdat de FAO functie alleen 
        de verhouding Tpot_jaar/Tpot_langjarig en Tact/Tpot gebruikt. 
    - Ky is de gewasresponsfactor (>1 betekent gevoelig voor transpiratiereductie)
    - De volgende Ky-parameters zijn gebruikt 
         1 gras                 1.11 (Gras_maaien, gekalibreerd met WOFOST)
         2 mais                 1.67 (Snijmais, gekalibreerd met WOFOST)
         3 aardappelen          1.12 (Aardappel_overig, gekalibreerd met WOFOST)
         4 bieten               1.82 (Suikerbieten, gekalibreerd met WOFOST)
         5 granen               1.45 (Zomergerst, gekalibreerd met FAO)
         6 overige landbouwgew  1.12 (Aardappel_overig, gekalibreerd met WOFOST)
         7 boomteelt            0.85 (Laanbomen, bron: WWL)
         8 glastuinbouw         
         9 boomgaard            0.85 (Laanbomen, bron: WWL)
        10 bollen               1.00 (Tulp en Lelie, bron: WWL)
        11 loofbos              
        12 naaldbos             
        13 moeras               
        14 duinvegetatie        
        15 kale grond           
        16 water                
        17 natuurlijk gras      1.00 (WUR, 2019)
        18 stedelijk grasland   1.00 (WUR, 2019)
        19 donker naaldbos      
        20 heidevegetatie       
        21 fruitkwekerijen      0.85 (Appelboom, bron: WWL)
        22 sportvelden          

    
"""

import datetime
from datetime import datetime as dt_datetime
from pathlib import Path

import imod
import numpy as np
import xarray as xr


def create_ky_grid(lgn: xr.DataArray) -> xr.DataArray:
    """Maak een grid met ky-factoren"""
    ky = xr.full_like(lgn, np.nan, dtype=float)
    ky = ky.where(lgn != 1, 1.11)  # Gras 1.11 (Gras_maaien, gekalibreerd met WOFOST)
    ky = ky.where(lgn != 2, 1.67)  # Mais 1.67 (Snijmais, Gekalibreerd met WOFOST)
    ky = ky.where(
        lgn != 3, 1.12
    )  # Aardappel 1.12 (Aardappel_overig, Gekalibreerd met WOFOST)
    ky = ky.where(lgn != 4, 1.82)  # Bieten 1.82 (Suikerbieten, Gekalibreerd met WOFOST)
    ky = ky.where(lgn != 5, 1.45)  # Granen 1.45 (Zomergerst, Gecalibreerd met WOFOST)
    ky = ky.where(
        lgn != 6, 1.12
    )  # Overig landbouwgewas (Aardappel, gekalibreerd met WOFOST)
    ky = ky.where(lgn != 7, 0.85)  # Boomteelt (Laanbomen, bron: WWL hoofdrapport, 2018)
    ky = ky.where(lgn != 9, 0.85)  # Boomgaard (Laanbomen, bron: WWL hoofdrapport, 2018)
    ky = ky.where(lgn != 10, 1.0)  # Bollen (Tulp en lelie, bron: FAO)
    ky = ky.where(lgn != 17, 1.0)  # Gras 1.00 (Bron: Hoving, et al., 2019. Schatten van
    # grasopbrengst op basis van spectrale reflectie, grashoogte en modellering)
    ky = ky.where(lgn != 18, 1.0)  # Gras 1.00 (Bron: Hoving, et al., 2019. Schatten van
    # grasopbrengst op basis van spectrale reflectie, grashoogte en modellering)
    ky = ky.where(
        lgn != 21, 0.85
    )  # Fruitkwekerijen (Appelboom, bron: WWL hoofdrapport, 2018)
    return ky


def create_potential_yield_grid(lgn: xr.DataArray) -> xr.DataArray:
    """Create a grid with long-term potential yield for FAO crops."""
    yldpot = xr.full_like(lgn, np.nan, dtype=float)

    yldpot = yldpot.where(lgn != 6, YLD_POT_OVERIG)
    yldpot = yldpot.where(lgn != 7, YLD_POT_BOOMTEELT)
    yldpot = yldpot.where(lgn != 9, YLD_POT_FRUIT)
    yldpot = yldpot.where(lgn != 10, YLD_POT_BOLLEN)
    yldpot = yldpot.where(lgn != 21, YLD_POT_FRUIT)
    return yldpot


def prepare_reference_tpot(
    tpot_files: list[Path],
    lgn: xr.DataArray,
    fao_lgn_classes: list[int],
) -> xr.DataArray:
    """
    Prepare long-term reference potential transpiration for FAO crops.

    The maximum potential transpiration is determined from the
    reference-period Tpot rasters. For each FAO crop class, the mean
    reference Tpot is calculated and assigned to all cells of that
    crop class.

    Parameters
    ----------
    tpot_files : list[Path]
        Paths to the reference Tpot raster files.
    lgn : xr.DataArray
        Land-use raster containing LGN crop classes.
    fao_lgn_classes : list[int]
        LGN classes for which FAO yields are calculated.

    Returns
    -------
    xr.DataArray
        Long-term reference potential transpiration per FAO crop class.
    """
    tpot_max_ref = xr.where(
        lgn > 0,
        0,
        np.nan,
    )

    tpot_ref = xr.concat(
        [imod.rasterio.open(path) for path in tpot_files],
        dim="time",
    )

    tpot_ref = xr.where(
        tpot_ref > 1e37,
        np.nan,
        tpot_ref,
    )

    tpot_max = tpot_ref.max(dim="time")

    tpot_max_ref = xr.where(
        tpot_max_ref < tpot_max,
        tpot_max_ref,
        tpot_max,
    )

    # Convert MetaSWAP flux sign convention to positive Tpot.
    tpot_langjarig = tpot_max_ref * -1

    tpot_langjarig_crop = xr.full_like(
        tpot_langjarig,
        np.nan,
    )

    for lgn_num in fao_lgn_classes:
        tpot_num = tpot_langjarig.where(lgn == lgn_num).mean(skipna=True)

        tpot_langjarig_crop = xr.where(
            lgn == lgn_num,
            tpot_num,
            tpot_langjarig_crop,
        )

    return tpot_langjarig_crop


def load_transpiration(
    files: list[Path],
) -> xr.DataArray:
    """Load and combine annual transpiration rasters."""
    data = xr.concat(
        [imod.rasterio.open(path) for path in files],
        dim="time",
    )

    data = xr.where(
        abs(data) > 1e37,
        np.nan,
        data,
    )

    return data


def calculate_fao_yields(
    startjaar: int,
    eindjaar: int,
    msw_tact: xr.DataArray,
    msw_tpot: xr.DataArray,
    tpot_langjarig: xr.DataArray,
    yldpot_langjarig: xr.DataArray,
    ky: xr.DataArray,
) -> xr.Dataset:
    """
    Calculate annual potential, relative, and actual crop yields.

    Potential yield is scaled using the ratio between annual and
    long-term potential transpiration. Relative yield is calculated
    from the FAO crop response factor Ky and the ratio of actual to
    potential transpiration. Actual yield is potential yield multiplied
    by relative yield.

    Parameters
    ----------
    startjaar : int
        First year of the calculation.
    eindjaar : int
        Last year of the calculation.
    msw_tact : xr.DataArray
        Actual MetaSWAP transpiration.
    msw_tpot : xr.DataArray
        Potential MetaSWAP transpiration.
    tpot_langjarig : xr.DataArray
        Long-term reference potential transpiration.
    yldpot_langjarig : xr.DataArray
        Long-term potential crop yield.
    ky : xr.DataArray
        FAO crop response factor.

    Returns
    -------
    xr.Dataset
        Annual potential, relative, and actual crop yields.
    """
    ## KY kaart in juiste extent zetten
    xmin = msw_tpot.x.min().values
    xmax = msw_tpot.x.max().values
    ymin = msw_tpot.y.min().values
    ymax = msw_tpot.y.max().values
    ky_ext = ky.sel(x=slice(xmin, xmax)).sel(y=slice(ymax, ymin))

    fao_mask = yldpot_langjarig.notnull()

    potential_yields = []
    relative_yields = []
    actual_yields = []
    jaren = range(startjaar, eindjaar + 1)
    for j in jaren:
        ## transpiratie voor specifiek jaar berekenen
        tpot_jaar = (
            msw_tpot.sel(time=slice(f"{j!s}-01-01", f"{j!s}-12-31")).sum(
                "time", min_count=1
            )
            * -1
        )

        tact_jaar = (
            msw_tact.sel(time=slice(f"{j!s}-01-01", f"{j!s}-12-31")).sum(
                "time", min_count=1
            )
            * -1
        )

        # Only calculate FAO yield where Tpot is valid
        valid_annual_tpot = tpot_jaar.notnull() & (tpot_jaar != 0)

        tpot_jaar = tpot_jaar.where(valid_annual_tpot)
        tact_jaar = tact_jaar.where(valid_annual_tpot)

        # Potential yield
        yldpot = (tpot_jaar / tpot_langjarig) * yldpot_langjarig

        yldpot["time"] = datetime.datetime(j, 1, 1)

        # Relative yield
        yldrel = 1 - ky_ext * (1 - tact_jaar / tpot_jaar)

        yldrel["time"] = datetime.datetime(j, 1, 1)

        # Actual yield
        yldact = yldpot * yldrel
        yldact["time"] = datetime.datetime(j, 1, 1)

        potential_yields.append(yldpot)
        relative_yields.append(yldrel)
        actual_yields.append(yldact)
        print(f"Processed year {j}")

    ## Voeg samen tot één data-array
    opbrengst = xr.Dataset()
    opbrengst["potentieel"] = xr.concat(potential_yields, dim="time").where(fao_mask)
    opbrengst["actueel"] = xr.concat(actual_yields, dim="time").where(fao_mask)
    opbrengst["relatief"] = xr.concat(relative_yields, dim="time").where(fao_mask)

    opbrengst = opbrengst.compute()
    return opbrengst


# Specificatie langjarige grids
"""
7	Boomteelt	kg ha-1	 10
9	Fruit	kg ha-1	     33628
10	Bollen	stuks ha-1	 303748
"""

YLD_POT_BOOMTEELT = 10
YLD_POT_FRUIT = 33628
YLD_POT_BOLLEN = 303748
YLD_POT_OVERIG = 8290

FAO_LGN_CLASSES = [6, 7, 9, 10, 21]


def main() -> None:
    start_time = dt_datetime.now()

    # Parameters
    startjaar = 2014
    eindjaar = 2015

    # Paths
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

    lgn = imod.idf.open(path_lgn)

    # # Specificatie langjarige grids
    # yldpot_langjarig = imod.idf.open(r'g:\LHM_modelruns\model_v4.3\40_jarige_run\runs\postprocessing\Droogteschade_FAO\pot_opbrengst_LHM4.3WOF_1980-2022_WOFOST.idf')
    # tpot_langjarig = imod.idf.open(r'g:\LHM_modelruns\model_v4.3\40_jarige_run\runs\postprocessing\Droogteschade_FAO\Tpot_LHM4.3WOF_1980-2022.idf')

    # Create FAO crop-specific grids
    ky = create_ky_grid(lgn)
    yldpot_langjarig = create_potential_yield_grid(lgn)

    ## Bereken Tpot per gewas per subregio
    tpot_langjarig = prepare_reference_tpot(tpot_files, lgn, FAO_LGN_CLASSES)

    valid_reference_tpot = tpot_langjarig.notnull() & (tpot_langjarig != 0)

    ## transpiratie inlezen
    msw_tpot = load_transpiration(tpot_files)
    msw_tact = load_transpiration(tact_files)

    # restrict msw_tact and msw_tpot to valid reference areas
    msw_tpot = msw_tpot.where(valid_reference_tpot)
    msw_tact = msw_tact.where(valid_reference_tpot)

    ## Opbrengst berekenen
    opbrengst = calculate_fao_yields(
        startjaar, eindjaar, msw_tact, msw_tpot, tpot_langjarig, yldpot_langjarig, ky
    )

    ## Wegschrijven
    imod.idf.save(
        path_output.joinpath("actuele_opbrengst.idf"),
        opbrengst["actueel"],
        pattern="{name}_{time.year}.idf",
    )
    imod.idf.save(
        path_output.joinpath("potentiele_opbrengst.idf"),
        opbrengst["potentieel"],
        pattern="{name}_{time.year}.idf",
    )
    imod.idf.save(
        path_output.joinpath("relatieve_opbrengst.idf"),
        opbrengst["relatief"],
        pattern="{name}_{time.year}.idf",
    )

    end_time = dt_datetime.now()
    print(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    main()
