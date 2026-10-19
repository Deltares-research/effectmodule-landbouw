import numpy as np

from src.utils import calculate_index, get_prijstool_option_runs


def test_calculate_index():
    scenario = np.array([10, 20, 30])
    bias = np.array([1, 2, 3])
    totaal = np.array([3, 6, 9])

    result = calculate_index(scenario, bias, totaal)

    expected = np.array([3, 3, 3])

    np.testing.assert_array_equal(result, expected)


def test_get_prijstool_option_runs_combiderv():
    result = get_prijstool_option_runs(
        "combiderv",
        ["REF2017BP18"],
    )

    assert result == ["REF2017BP18_totaal_combiderv_optdervderv_ActMed_new"]


def test_get_prijstool_option_runs_wof():
    result = get_prijstool_option_runs(
        "wof",
        ["REF2017BP18"],
    )

    assert len(result) == 5
