from pathlib import Path

import imod


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


    pass

if __name__ == "__main__":
    main()