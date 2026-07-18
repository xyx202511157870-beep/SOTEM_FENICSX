from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.constants import mu_0
from atem3d.sotem_observables import CanonicalResponse, canonical_response


CANONICAL_COLUMNS = (
    "Ex_V_per_m",
    "Ey_V_per_m",
    "Hz_A_per_m",
    "Bz_T",
    "dBzdt_T_per_s",
)


def _direct_response(times=None, values=None, columns=CANONICAL_COLUMNS):
    if times is None:
        times = np.array([1.0e-5, 1.0e-4])
    if values is None:
        values = np.array(
            [
                [2.0, 1.0e-4, 3.0, mu_0 * 3.0, 4.0],
                [1.0, 5.0e-5, 2.0, mu_0 * 2.0, 3.0],
            ]
        )
    return CanonicalResponse(times=times, values=values, columns=columns)


def test_canonical_response_writes_hz_bz_and_dbdt_without_aliasing():
    table = canonical_response(
        np.array([1.0e-5, 1.0e-4]),
        np.array([[2.0, 1.0e-4, 3.0, 4.0], [1.0, 5.0e-5, 2.0, 3.0]]),
        ["Ex", "Ey", "Hz", "dBzdt"],
    )
    assert table.columns == CANONICAL_COLUMNS
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
    "columns",
    [
        tuple(reversed(CANONICAL_COLUMNS)),
        (*CANONICAL_COLUMNS[:-1], "dBzdt_nT_per_s"),
        CANONICAL_COLUMNS[:-1],
        list(CANONICAL_COLUMNS),
    ],
)
def test_direct_construction_requires_exact_canonical_column_tuple(columns):
    with pytest.raises(
        ValueError,
        match="columns must equal the canonical response columns",
    ):
        _direct_response(columns=columns)


@pytest.mark.parametrize(
    ("times", "row_count"),
    [
        (np.empty((0,), dtype=float), 0),
        (np.array([[1.0e-5, 1.0e-4]]), 1),
        (np.array([1.0e-5, np.nan]), 2),
        (np.array([1.0e-5, np.inf]), 2),
        (np.array([0.0, 1.0e-4]), 2),
        (np.array([-1.0e-5, 1.0e-4]), 2),
        (np.array([1.0e-5, 1.0e-5]), 2),
        (np.array([1.0e-4, 1.0e-5]), 2),
    ],
)
def test_direct_construction_rejects_invalid_times(times, row_count):
    with pytest.raises(ValueError):
        _direct_response(times=times, values=np.zeros((row_count, 5)))


@pytest.mark.parametrize(
    "values",
    [
        np.zeros(5),
        np.zeros((1, 1, 5)),
        np.zeros((1, 5)),
        np.zeros((2, 4)),
        np.zeros((2, 6)),
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, np.nan, 4.0, 5.0],
            ]
        ),
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.0, 2.0, np.inf, 4.0, 5.0],
            ]
        ),
    ],
)
def test_direct_construction_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        _direct_response(values=values)


def test_valid_direct_construction_copies_and_deeply_freezes_arrays():
    times = np.array([1.0e-5, 1.0e-4])
    values = np.array(
        [
            [2.0, 1.0e-4, 3.0, mu_0 * 3.0, 4.0],
            [1.0, 5.0e-5, 2.0, mu_0 * 2.0, 3.0],
        ]
    )

    table = _direct_response(times=times, values=values)

    assert not np.shares_memory(table.times, times)
    assert not np.shares_memory(table.values, values)
    assert table.times.dtype == np.float64
    assert table.values.dtype == np.float64
    with pytest.raises(ValueError):
        table.times[0] = 2.0e-5
    with pytest.raises(ValueError):
        table.values[0, 0] = 10.0
    with pytest.raises(ValueError):
        table.times.setflags(write=True)
    with pytest.raises(ValueError):
        table.values.setflags(write=True)


def test_direct_construction_rejects_inconsistent_hz_bz_relation():
    values = np.array([[1.0, 2.0, 3.0, 999.0, 4.0]])

    with pytest.raises(
        ValueError,
        match=r"^Bz_T must equal mu_0 \* Hz_A_per_m$",
    ):
        _direct_response(times=np.array([1.0e-5]), values=values)


def test_direct_construction_rejects_nonzero_bz_when_hz_is_zero():
    smallest_positive = np.nextafter(0.0, 1.0)
    values = np.array([[1.0, 2.0, 0.0, smallest_positive, 4.0]])

    with pytest.raises(
        ValueError,
        match=r"^Bz_T must equal mu_0 \* Hz_A_per_m$",
    ):
        _direct_response(times=np.array([1.0e-5]), values=values)


def test_direct_construction_rejects_opposite_hz_bz_signs():
    values = np.array([[1.0, 2.0, -3.0, mu_0 * 3.0, 4.0]])

    with pytest.raises(
        ValueError,
        match=r"^Bz_T must equal mu_0 \* Hz_A_per_m$",
    ):
        _direct_response(times=np.array([1.0e-5]), values=values)


@pytest.mark.parametrize("hz", [3.0, -3.0])
def test_direct_construction_accepts_exact_signed_hz_bz_relation(hz):
    values = np.array([[1.0, 2.0, hz, mu_0 * hz, 4.0]])

    table = _direct_response(times=np.array([1.0e-5]), values=values)

    assert table.values[0, 2] == hz
    assert table.values[0, 3] == mu_0 * hz


def test_factory_preserves_exact_signed_hz_bz_relation():
    table = canonical_response(
        [1.0e-5, 1.0e-4],
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, -3.0, 4.0]],
        ["Ex", "Ey", "Hz", "dBzdt"],
    )

    np.testing.assert_array_equal(table.values[:, 3], mu_0 * table.values[:, 2])


def test_canonical_response_equality_is_identity_based():
    first = _direct_response()
    second = _direct_response()

    assert first == first
    assert first != second


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


def test_max_finite_hz_keeps_bz_finite():
    max_float = np.finfo(np.float64).max

    table = canonical_response(
        [1.0e-5],
        [[1.0, 1.0, max_float, 1.0]],
        ["Ex", "Ey", "Hz", "dBzdt"],
    )

    assert table.values[0, 2] == max_float
    assert np.isfinite(table.values[0, 3])
    assert table.values[0, 3] == mu_0 * max_float


def test_peak_ratio_overflow_returns_inf_without_nan():
    table = canonical_response(
        [1.0e-5],
        [[np.finfo(np.float64).tiny, np.finfo(np.float64).max, 1.0, 1.0]],
        ["Ex", "Ey", "Hz", "dBzdt"],
    )

    assert np.isinf(table.ey_to_ex_peak_ratio)
    assert not np.isnan(table.ey_to_ex_peak_ratio)


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

    raw = path.read_bytes()
    assert b"\r\n" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 3
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "time_obs_s,Ex_V_per_m,Ey_V_per_m,Hz_A_per_m,Bz_T,dBzdt_T_per_s"
    )
    loaded = np.loadtxt(path, delimiter=",", skiprows=1)
    np.testing.assert_array_equal(loaded, np.column_stack((table.times, table.values)))
    reconstructed = CanonicalResponse(
        times=loaded[:, 0],
        values=loaded[:, 1:],
        columns=CANONICAL_COLUMNS,
    )
    np.testing.assert_array_equal(reconstructed.values, table.values)


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
