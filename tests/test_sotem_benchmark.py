from pathlib import Path
import re

import numpy as np
import pytest
import yaml

from atem3d.sotem_benchmark import load_benchmark_case


ROOT = Path(__file__).resolve().parents[1]


def _valid_payload():
    return {
        "case_id": "test_case",
        "coordinates": "z_down",
        "source": {
            "start_m": [-500.0, 0.0, 0.1],
            "end_m": [500.0, 0.0, 0.1],
            "current_a": 1.0,
            "waveform": "ideal_step_off",
        },
        "receiver": {"location_m": [0.0, 800.0, 0.1]},
        "earth": {"rho_ohm_m": 100.0},
        "air": {"rho_ohm_m": 1.0e8},
        "times": {
            "kind": "logspace",
            "start_s": 1.0e-5,
            "stop_s": 1.0e-1,
            "count": 41,
        },
        "components": ["Ex", "Ey", "Hz", "dBzdt"],
    }


def _valid_layered_earth():
    return {
        "layers": [
            {"top_m": 0.0, "bottom_m": 300.0, "rho_ohm_m": 100.0},
            {"top_m": 300.0, "bottom_m": None, "rho_ohm_m": 100.0},
        ]
    }


def _load_payload(tmp_path, payload):
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_benchmark_case(path)


def test_lei_case_matches_approved_design():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/lei2023_noip.yaml")
    assert case.case_id == "lei2023_noip"
    assert case.coordinates == "z_down"
    assert case.source_start_down == (-500.0, 0.0, 0.1)
    assert case.source_end_down == (500.0, 0.0, 0.1)
    assert case.receiver_down == (0.0, 800.0, 0.1)
    assert case.current_a == 1.0
    assert case.rho_air_ohm_m == 1.0e8
    assert case.earth == {"rho_ohm_m": 100.0}
    assert case.polarization is None
    assert case.components == ("Ex", "Ey", "Hz", "dBzdt")
    assert case.observation_times.size == 41
    assert case.observation_times[[0, -1]].tolist() == pytest.approx(
        [1.0e-5, 1.0e-1]
    )


def test_song_pair_changes_only_polarization():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")
    assert case.case_id == "song2025_layered_pair"
    assert case.coordinates == "z_down"
    assert case.source_start_down == (-500.0, 0.0, 0.1)
    assert case.source_end_down == (500.0, 0.0, 0.1)
    assert case.receiver_down == (0.0, -500.0, 0.1)
    assert case.current_a == 10.0
    assert case.rho_air_ohm_m == 1.0e6
    assert case.earth == {
        "layers": (
            {"top_m": 0.0, "bottom_m": 300.0, "rho_ohm_m": 100.0},
            {"top_m": 300.0, "bottom_m": None, "rho_ohm_m": 100.0},
        )
    }
    assert case.components == ("Ex", "Ey", "Hz", "dBzdt")
    assert case.observation_times.size == 51
    assert case.observation_times[[0, -1]].tolist() == pytest.approx(
        [1.0e-5, 1.0]
    )
    assert case.polarization == {
        "top_m": 0.0,
        "bottom_m": 300.0,
        "rho0_ohm_m": 100.0,
        "m": 0.3,
        "tau_s": 1.0,
        "c": 0.3,
    }


def test_loader_rejects_non_mapping_root(tmp_path):
    with pytest.raises(ValueError):
        _load_payload(tmp_path, ["not", "a", "mapping"])


def test_loaded_nested_model_data_is_immutable():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")

    with pytest.raises(TypeError):
        case.earth["layers"][0]["rho_ohm_m"] = 50.0
    with pytest.raises(TypeError):
        case.polarization["m"] = 0.5
    with pytest.raises(TypeError):
        case.components[0] = "Ey"


def test_loaded_observation_times_cannot_be_modified_or_made_writeable():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/lei2023_noip.yaml")

    with pytest.raises(ValueError):
        case.observation_times[0] = 2.0e-5
    with pytest.raises(ValueError):
        case.observation_times.setflags(write=True)


@pytest.mark.parametrize("invalid_case_id", ["", "   ", 123, True])
def test_loader_rejects_invalid_case_id(tmp_path, invalid_case_id):
    payload = _valid_payload()
    payload["case_id"] = invalid_case_id

    with pytest.raises(ValueError, match="^case_id must be a non-empty string$"):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize("field", ["source", "receiver", "air", "times"])
def test_loader_rejects_non_mapping_nested_sections(tmp_path, field):
    payload = _valid_payload()
    payload[field] = []

    with pytest.raises(ValueError, match=rf"^{field} must be a mapping$"):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("source", "current_a"),
        ("receiver", "location_m"),
        ("air", "rho_ohm_m"),
        ("times", "start_s"),
    ],
)
def test_loader_reports_missing_nested_fields(tmp_path, section, field):
    payload = _valid_payload()
    del payload[section][field]

    with pytest.raises(ValueError, match=rf"^{section}\.{field} is required$"):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize("invalid_rho", [-1.0, 0.0, np.nan, True, "100.0"])
def test_loader_rejects_invalid_halfspace_resistivity(tmp_path, invalid_rho):
    payload = _valid_payload()
    payload["earth"]["rho_ohm_m"] = invalid_rho

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize("invalid_rho", [-1.0, 0.0, np.nan, False, "100.0"])
def test_loader_rejects_invalid_layer_resistivity(tmp_path, invalid_rho):
    payload = _valid_payload()
    payload["earth"] = _valid_layered_earth()
    payload["earth"]["layers"][0]["rho_ohm_m"] = invalid_rho

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "invalid_depth"),
    [
        ("top_m", np.nan),
        ("top_m", True),
        ("top_m", "0.0"),
        ("bottom_m", np.inf),
        ("bottom_m", False),
        ("bottom_m", "300.0"),
    ],
)
def test_loader_rejects_invalid_layer_depths(tmp_path, field, invalid_depth):
    payload = _valid_payload()
    payload["earth"] = _valid_layered_earth()
    payload["earth"]["layers"][0][field] = invalid_depth

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "layers",
    [
        [
            {"top_m": 0.0, "bottom_m": 0.0, "rho_ohm_m": 100.0},
        ],
        [
            {"top_m": 0.0, "bottom_m": 300.0, "rho_ohm_m": 100.0},
            {"top_m": 301.0, "bottom_m": None, "rho_ohm_m": 100.0},
        ],
        [
            {"top_m": 0.0, "bottom_m": 300.0, "rho_ohm_m": 100.0},
            {"top_m": 299.0, "bottom_m": None, "rho_ohm_m": 100.0},
        ],
        [
            {"top_m": 0.0, "bottom_m": None, "rho_ohm_m": 100.0},
            {"top_m": 300.0, "bottom_m": None, "rho_ohm_m": 100.0},
        ],
    ],
)
def test_loader_rejects_invalid_layer_order_or_continuity(tmp_path, layers):
    payload = _valid_payload()
    payload["earth"] = {"layers": layers}

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "earth",
    [
        {"rho_ohm_m": 100.0, "extra": 1.0},
        {"rho_ohm_m": 100.0, "layers": []},
        {"layers": [], "extra": 1.0},
        {},
    ],
)
def test_loader_rejects_unknown_or_mixed_earth_schema(tmp_path, earth):
    payload = _valid_payload()
    payload["earth"] = earth

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize("mutate_layer", ["extra", "missing"])
def test_loader_requires_exact_layer_fields(tmp_path, mutate_layer):
    payload = _valid_payload()
    payload["earth"] = _valid_layered_earth()
    if mutate_layer == "extra":
        payload["earth"]["layers"][0]["extra"] = 1.0
    else:
        del payload["earth"]["layers"][0]["rho_ohm_m"]

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("coordinates", "z_up"),
        ("waveform", "ramp_off"),
        ("components", ["Ex", "Ey", "dBzdt", "Hz"]),
    ],
)
def test_loader_rejects_unapproved_case_conventions(tmp_path, field, invalid_value):
    payload = _valid_payload()
    if field == "waveform":
        payload["source"][field] = invalid_value
    else:
        payload[field] = invalid_value

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    "invalid_times",
    [
        {"kind": "linear", "start_s": 1.0e-5, "stop_s": 1.0e-1, "count": 41},
        {"kind": "logspace", "start_s": 0.0, "stop_s": 1.0e-1, "count": 41},
        {"kind": "logspace", "start_s": 1.0e-1, "stop_s": 1.0e-1, "count": 41},
        {"kind": "logspace", "start_s": 1.0e-5, "stop_s": 1.0e-1, "count": 1},
        {"kind": "logspace", "start_s": 1.0e-5, "stop_s": np.inf, "count": 41},
        {"kind": "logspace", "start_s": "1.0e-5", "stop_s": 1.0e-1, "count": 41},
        {"kind": "logspace", "start_s": 1.0e-5, "stop_s": "1.0", "count": 41},
        {"kind": "logspace", "start_s": 1.0e-5, "stop_s": 1.0e-1, "count": True},
        {"kind": "logspace", "start_s": 1.0e-5, "stop_s": 1.0e-1, "count": 41.0},
    ],
)
def test_loader_rejects_invalid_time_definitions(tmp_path, invalid_times):
    payload = _valid_payload()
    payload["times"] = invalid_times

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("section", "field", "invalid_value", "name"),
    [
        ("source", "start_m", [0.0, 1.0], "source.start_m"),
        ("source", "end_m", [0.0, 1.0, 2.0, 3.0], "source.end_m"),
        ("receiver", "location_m", [0.0, np.nan, 0.1], "receiver.location_m"),
        ("source", "start_m", "123", "source.start_m"),
        ("source", "start_m", [True, 0.0, 0.1], "source.start_m"),
        ("source", "start_m", ["-500.0", 0.0, 0.1], "source.start_m"),
    ],
)
def test_loader_rejects_invalid_vectors(tmp_path, section, field, invalid_value, name):
    payload = _valid_payload()
    payload[section][field] = invalid_value

    with pytest.raises(
        ValueError, match=rf"^{re.escape(name)} must contain three finite values$"
    ):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("source", "current_a", 0.0),
        ("source", "current_a", np.inf),
        ("source", "current_a", True),
        ("source", "current_a", "1.0"),
        ("air", "rho_ohm_m", -1.0),
        ("air", "rho_ohm_m", np.nan),
        ("air", "rho_ohm_m", False),
        ("air", "rho_ohm_m", "100000000.0"),
    ],
)
def test_loader_rejects_nonpositive_or_nonfinite_scalars(
    tmp_path, section, field, invalid_value
):
    payload = _valid_payload()
    payload[section][field] = invalid_value

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("m", -0.1),
        ("m", 1.0),
        ("tau_s", 0.0),
        ("c", 0.0),
        ("c", 1.1),
        ("bottom_m", 0.0),
        ("rho0_ohm_m", 0.0),
    ],
)
def test_loader_rejects_invalid_polarization_bounds(tmp_path, field, invalid_value):
    payload = _valid_payload()
    payload["polarization"] = {
        "top_m": 0.0,
        "bottom_m": 300.0,
        "rho0_ohm_m": 100.0,
        "m": 0.3,
        "tau_s": 1.0,
        "c": 0.3,
    }
    payload["polarization"][field] = invalid_value

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("top_m", "0.0"),
        ("rho0_ohm_m", "100.0"),
        ("tau_s", True),
        ("c", False),
    ],
)
def test_loader_rejects_non_numeric_polarization_values(
    tmp_path, field, invalid_value
):
    payload = _valid_payload()
    payload["polarization"] = {
        "top_m": 0.0,
        "bottom_m": 300.0,
        "rho0_ohm_m": 100.0,
        "m": 0.3,
        "tau_s": 1.0,
        "c": 0.3,
    }
    payload["polarization"][field] = invalid_value

    with pytest.raises(ValueError):
        _load_payload(tmp_path, payload)


def test_loader_normalizes_integer_polarization_values_to_float(tmp_path):
    payload = _valid_payload()
    payload["polarization"] = {
        "top_m": 0,
        "bottom_m": 300,
        "rho0_ohm_m": 100,
        "m": 0,
        "tau_s": 1,
        "c": 1,
    }

    case = _load_payload(tmp_path, payload)

    assert case.polarization == {
        "top_m": 0.0,
        "bottom_m": 300.0,
        "rho0_ohm_m": 100.0,
        "m": 0.0,
        "tau_s": 1.0,
        "c": 1.0,
    }
    assert all(type(value) is float for value in case.polarization.values())
