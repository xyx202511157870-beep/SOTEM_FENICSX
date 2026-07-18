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


def _load_payload(tmp_path, payload):
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return load_benchmark_case(path)


def test_lei_case_matches_approved_design():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/lei2023_noip.yaml")
    assert case.source_start_down == (-500.0, 0.0, 0.1)
    assert case.source_end_down == (500.0, 0.0, 0.1)
    assert case.receiver_down == (0.0, 800.0, 0.1)
    assert case.current_a == 1.0
    assert case.rho_air_ohm_m == 1.0e8
    assert case.observation_times.size == 41
    assert case.observation_times[[0, -1]].tolist() == pytest.approx(
        [1.0e-5, 1.0e-1]
    )


def test_song_pair_changes_only_polarization():
    case = load_benchmark_case(ROOT / "benchmarks/sotem/song2025_layered_pair.yaml")
    assert case.source_start_down == (-500.0, 0.0, 0.1)
    assert case.source_end_down == (500.0, 0.0, 0.1)
    assert case.receiver_down == (0.0, -500.0, 0.1)
    assert case.current_a == 10.0
    assert case.rho_air_ohm_m == 1.0e6
    assert case.observation_times.size == 51
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
        ("air", "rho_ohm_m", -1.0),
        ("air", "rho_ohm_m", np.nan),
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
