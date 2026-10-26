import numpy as np


def calculate_yield_loss(cropdata, method):
    """
    Calculate yield loss according to the selected method.

    Parameters
    ----------
    cropdata : pd.DataFrame
        Crop database containing yield information.

    method : str
        Yield loss calculation method.

    Returns
    -------
    pd.DataFrame
        Crop database with an added or updated
        'derv_opbrengst' column.
    """
    cropdata = cropdata.copy()

    # calculate derving given certain calculation method specified in method
    if method == "optdervActPot":
        cropdata["derv_opbrengst"] = (
            cropdata["pot_opbrengst"] - cropdata["act_opbrengst"]
        )

    elif method == "optdervAct_Actgem":
        cropdata["derv_opbrengst"] = (
            cropdata["gem_opbrengst"] - cropdata["act_opbrengst"]
        )

    # MvdV 250128 optie aangepast om derving te bepalen ten opzicht van het minimum van potentiele en mediane opbrengst
    elif method == "optdervAct_Actmed":
        cropdata["derv_opbrengst"] = (
            cropdata["med_opbrengst"] - cropdata["act_opbrengst"]
        )

    elif method == "optdervAct_ActMedPot":
        cropdata["derv_opbrengst"] = np.where(
            cropdata["med_opbrengst"] < cropdata["pot_opbrengst"],
            cropdata["med_opbrengst"] - cropdata["act_opbrengst"],
            cropdata["pot_opbrengst"] - cropdata["act_opbrengst"],
        )

    elif method == "optdervderv_ActMed_new":
        pot_act = cropdata["pot_opbrengst"] - cropdata["act_opbrengst"]
        cropdata["derv_opbrengst"] = pot_act - cropdata["ref_med_derv"]

        # real
        med_act_opbrengst = cropdata.groupby(["regio", "landgebruik"])[
            "act_opbrengst"
        ].median()
        cropdata["damage_calculation_yield"] = [
            med_act_opbrengst[r, l] - c
            for r, l, c in zip(
                cropdata.index.get_level_values("regio"),
                cropdata.index.get_level_values("landgebruik"),
                cropdata["derv_opbrengst"],
            )
        ]
    else:
        raise ValueError(f"Unknown calculation method: {method}")

    if method != "optdervActPot":
        cropdata["derv_opbrengst"] = cropdata["derv_opbrengst"].clip(lower=0)

    return cropdata
