from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.constants import mu_0
from atem3d.sotem_observables import canonical_response


def test_canonical_response_writes_hz_bz_and_dbdt_without_aliasing():
    table = canonical_response(
        np.array([1.0e-5, 1.0e-4]),
        np.array([[2.0, 1.0e-4, 3.0, 4.0], [1.0, 5.0e-5, 2.0, 3.0]]),
        ["Ex", "Ey", "Hz", "dBzdt"],
    )
    assert table.columns == (
        "Ex_V_per_m",
        "Ey_V_per_m",
        "Hz_A_per_m",
        "Bz_T",
        "dBzdt_T_per_s",
    )
    np.testing.assert_allclose(table.values[:, 3], mu_0 * table.values[:, 2])
    assert table.ey_to_ex_peak_ratio == 5.0e-5


def test_canonical_response_uses_canonical_order_and_ignores_diagnostics():
    table = canonical_response(
        [1.0e-5, 1.0e-4],
        [
            [99.0, 4.0, -1.0e-4, 3.0, -2.0],
            [98.0, 3.0, 5.0e-5, 2.0, 1.0],
        ],
        ["diagnostic", "dBzdt", "Ey", "Hz", "Ex"],
    )

    np.testing.assert_allclose(
        table.values,
        [
            [-2.0, -1.0e-4, 3.0, mu_0 * 3.0, 4.0],
            [1.0, 5.0e-5, 2.0, mu_0 * 2.0, 3.0],
        ],
    )
    assert table.ey_to_ex_peak_ratio == 5.0e-5


def test_canonical_response_accepts_component_iterators():
    table = canonical_response(
        [1.0e-5],
        [[1.0, 2.0, 3.0, 4.0]],
        iter(["Ex", "Ey", "Hz", "dBzdt"]),
    )

    assert table.values.shape == (1, 5)


def test_canonical_response_reports_missing_components_in_requested_order():
    with pytest.raises(
        ValueError,
        match=r"^missing canonical components: Ex, Ey, dBzdt$",
    ):
        canonical_response([1.0e-5], [[3.0, 9.0]], ["Hz", "diagnostic"])


@pytest.mark.parametrize("duplicate", ["Ex", "Ey", "Hz", "dBzdt"])
def test_canonical_response_rejects_duplicate_required_components(duplicate):
    components = ["Ex", "Ey", "Hz", "dBzdt", duplicate]

    with pytest.raises(ValueError, match=rf"duplicate canonical component: {duplicate}"):
        canonical_response([1.0e-5], [[1.0, 2.0, 3.0, 4.0, 5.0]], components)


@pytest.mark.parametrize(
    "components",
    [
        "ExEyHzdBzdt",
        [["Ex"], ["Ey"], ["Hz"], ["dBzdt"]],
        ["Ex", "Ey", "Hz", 1],
        123,
    ],
)
def test_canonical_response_rejects_invalid_component_collections(components):
    with pytest.raises(ValueError, match="components must be a one-dimensional iterable of strings"):
        canonical_response([1.0e-5], [[1.0, 2.0, 3.0, 4.0]], components)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [[1.0e-5, 1.0e-4]],
        [0.0, 1.0e-4],
        [-1.0e-5, 1.0e-4],
        [1.0e-5, np.nan],
        [1.0e-5, np.inf],
        [1.0e-5, 1.0e-5],
        [1.0e-4, 1.0e-5],
        [1.0e-5 + 0.0j, 1.0e-4 + 1.0j],
        [True, False],
        ["1e-5", "1e-4"],
    ],
)
def test_canonical_response_rejects_invalid_times(times):
    with pytest.raises(ValueError):
        canonical_response(times, np.empty((len(times), 4)), ["Ex", "Ey", "Hz", "dBzdt"])


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0, 4.0],
        [[[1.0, 2.0, 3.0, 4.0]]],
        [[1.0, 2.0, 3.0]],
        [[1.0, 2.0, 3.0, 4.0, 5.0]],
        [[1.0, 2.0, np.nan, 4.0]],
        [[1.0, 2.0, np.inf, 4.0]],
        [[1.0, 2.0, 3.0 + 1.0j, 4.0]],
        [[True, False, True, False]],
        [["1", "2", "3", "4"]],
    ],
)
def test_canonical_response_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        canonical_response([1.0e-5], values, ["Ex", "Ey", "Hz", "dBzdt"])


def test_canonical_response_requires_one_value_row_per_time():
    with pytest.raises(ValueError, match="values row count must equal times length"):
        canonical_response(
            [1.0e-5, 1.0e-4],
            [[1.0, 2.0, 3.0, 4.0]],
            ["Ex", "Ey", "Hz", "dBzdt"],
        )


def test_canonical_response_copies_and_deeply_freezes_scientific_arrays():
    input_times = np.array([1.0e-5, 1.0e-4])
    input_values = np.array([[2.0, 1.0e-4, 3.0, 4.0], [1.0, 5.0e-5, 2.0, 3.0]])

    table = canonical_response(
        input_times,
        input_values,
        ["Ex", "Ey", "Hz", "dBzdt"],
    )

    assert not np.shares_memory(table.times, input_times)
    assert not np.shares_memory(table.values, input_values)
    with pytest.raises(ValueError):
        table.times[0] = 2.0e-5
    with pytest.raises(ValueError):
        table.values[0, 0] = 10.0
    with pytest.raises(ValueError):
        table.times.setflags(write=True)
    with pytest.raises(ValueError):
        table.values.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        table.columns = ()


@pytest.mark.parametrize(
    ("ex", "ey", "expected"),
    [
        ([0.0, 0.0], [1.0, -2.0], float("inf")),
        ([0.0, 0.0], [0.0, 0.0], 0.0),
        ([-2.0, 1.0], [-4.0, 3.0], 2.0),
    ],
)
def test_ey_to_ex_peak_ratio_is_defined_without_nan(ex, ey, expected):
    table = canonical_response(
        [1.0e-5, 1.0e-4],
        np.column_stack((ex, ey, [3.0, 2.0], [4.0, 3.0])),
        ["Ex", "Ey", "Hz", "dBzdt"],
    )

    assert not np.isnan(table.ey_to_ex_peak_ratio)
    assert table.ey_to_ex_peak_ratio == expected


def test_write_csv_overwrites_with_exact_header_and_round_trips(tmp_path):
    path = tmp_path / "response.csv"
    path.write_text("stale content", encoding="utf-8")
    table = canonical_response(
        [1.0e-5, 0.12345678901234566],
        [
            [1.2345678901234567, -2.345678901234567, 3.456789012345678, -4.567890123456789],
            [-5.67890123456789, 6.789012345678901, -7.890123456789012, 8.901234567890123],
        ],
        ["Ex", "Ey", "Hz", "dBzdt"],
    )

    table.write_csv(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "time_obs_s,Ex_V_per_m,Ey_V_per_m,Hz_A_per_m,Bz_T,dBzdt_T_per_s"
    )
    loaded = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(loaded, np.column_stack((table.times, table.values)))


def test_write_csv_does_not_create_parent_directories(tmp_path):
    table = canonical_response(
        [1.0e-5],
        [[1.0, 2.0, 3.0, 4.0]],
        ["Ex", "Ey", "Hz", "dBzdt"],
    )
    path = tmp_path / "missing" / "response.csv"

    with pytest.raises(FileNotFoundError):
        table.write_csv(path)
    assert not path.parent.exists()
