import numpy as np
import pytest
import xarray as xr

from src.processing.prijstool.yields import get_standard_yield


def test_get_standard_yield_potential():
    cropdata = xr.Dataset(
        {
            "pot_opbrengst": (
                "Year",
                [100.0, 120.0, 140.0],
            ),
            "gem_opbrengst": (
                "Year",
                [80.0, 90.0, 100.0],
            ),
            "act_opbrengst": (
                "Year",
                [60.0, 80.0, 100.0],
            ),
        },
        coords={
            "Year": [2000, 2001, 2002],
        },
    )

    result = get_standard_yield(
        cropdata,
        method="optdervActPot",
    )

    expected = np.array([100.0, 120.0, 140.0])

    np.testing.assert_array_equal(
        result.values,
        expected,
    )


def test_get_standard_yield_average():
    cropdata = xr.Dataset(
        {
            "pot_opbrengst": (
                "Year",
                [100.0, 120.0, 140.0],
            ),
            "gem_opbrengst": (
                "Year",
                [80.0, 90.0, 100.0],
            ),
            "act_opbrengst": (
                "Year",
                [60.0, 80.0, 100.0],
            ),
        },
        coords={
            "Year": [2000, 2001, 2002],
        },
    )

    result = get_standard_yield(
        cropdata,
        method="optdervAct_Actgem",
    )

    expected = np.array([80.0, 90.0, 100.0])

    np.testing.assert_array_equal(
        result.values,
        expected,
    )


@pytest.mark.parametrize(
    "method",
    [
        "optdervAct_Actmed",
        "optdervderv_ActMed",
        "optdervderv_ActMed_new",
    ],
)
def test_get_standard_yield_median(method):
    cropdata = xr.Dataset(
        {
            "act_opbrengst": (
                "Year",
                [60.0, 100.0, 80.0],
            ),
        },
        coords={
            "Year": [2000, 2001, 2002],
        },
    )

    result = get_standard_yield(
        cropdata,
        method=method,
    )

    assert result.item() == 80.0


def test_get_standard_yield_unknown_method():
    cropdata = xr.Dataset()

    with pytest.raises(
        ValueError,
        match="Unknown calculation method",
    ):
        get_standard_yield(
            cropdata,
            method="typo",
        )
