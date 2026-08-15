from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from atem3d.empymod_compare import EmpymodSurvey
from atem3d.empymod_magnetic6 import (
    MAGNETIC6_COMPONENTS,
    MagneticSixReferenceResult,
)
from atem3d.empymod_magnetic6_cli import build_parser
from atem3d.empymod_waveform import (
    PiecewiseLinearTurnOff,
    load_turnoff_csv,
    run_empymod_magnetic6_waveform_reference,
    turnoff_waveform_from_config,
)


def _survey(times):
    return EmpymodSurvey(
        source_start=(-20.0, -7.0, 0.1),
        source_end=(20.0, 7.0, 0.1),
        receiver_locations=((13.0, 31.0, 0.2),),
        components=MAGNETIC6_COMPONENTS,
        times=np.asarray(times, dtype=float),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        strength=1.0,
        signal=-1,
        coordinate_system="depth_down",
    )


def _exponential_reference_runner(survey, **_kwargs):
    times = np.asarray(survey.times, dtype=float)
    component_scale = np.arange(1.0, 7.0)
    data = np.exp(-times[:, None, None]) * component_scale[None, None, :]
    dbdt = data[..., 3:].copy()
    return MagneticSixReferenceResult(
        times=times,
        receiver_locations=tuple(survey.receiver_locations),
        data=data,
        dbdt_native=dbdt.copy(),
        dbdt_impulse=dbdt.copy(),
        primary_dbdt_reference="native_b",
        empymod_version="2.6.0",
        audit={
            "empymod_version": "2.6.0",
            "primary_dbdt_reference": "native_b",
            "native_b_available": True,
            "requested": True,
            "tolerance": 1.0e-12,
            "floor_fraction": 0.01,
            "performed": True,
            "passed": True,
        },
    )


def test_linear_turnoff_metadata_and_current_drop_conservation():
    waveform = PiecewiseLinearTurnOff.linear(5.0e-6)

    assert waveform.duration == pytest.approx(5.0e-6)
    assert waveform.total_drop == pytest.approx(1.0)
    np.testing.assert_allclose(waveform.times, [-5.0e-6, 0.0])
    np.testing.assert_allclose(waveform.values, [1.0, 0.0])
    assert waveform.metadata()["time_origin"] == "ramp_end"


def test_finite_linear_ramp_matches_analytic_average_of_stepoff_response():
    duration = 0.2
    times = np.array([0.1, 0.7, 1.3])
    waveform = PiecewiseLinearTurnOff.linear(duration)

    result = run_empymod_magnetic6_waveform_reference(
        _survey(times),
        waveform,
        quadrature_order=8,
        reference_runner=_exponential_reference_runner,
    )

    expected_scalar = (
        np.exp(-times) - np.exp(-(times + duration))
    ) / duration
    expected = expected_scalar[:, None, None] * np.arange(1.0, 7.0)[None, None, :]
    np.testing.assert_allclose(result.data, expected, rtol=1.0e-12, atol=1.0e-14)
    np.testing.assert_allclose(result.dbdt_native, expected[..., 3:])
    np.testing.assert_allclose(result.dbdt_impulse, expected[..., 3:])
    assert result.convolution["weight_sum"] == pytest.approx(1.0)
    assert result.audit["performed"] is True
    assert result.audit["passed"] is True


def test_piecewise_linear_superposition_preserves_constant_response():
    waveform = PiecewiseLinearTurnOff(
        times=np.array([-3.0, -2.0, -0.5, 0.0]),
        values=np.array([1.0, 0.8, 0.2, 0.0]),
    )

    def constant_runner(survey, **_kwargs):
        shape = (len(survey.times), len(survey.receiver_locations), 6)
        data = np.full(shape, 4.0)
        dbdt = np.full(shape[:-1] + (3,), 4.0)
        return MagneticSixReferenceResult(
            times=np.asarray(survey.times),
            receiver_locations=tuple(survey.receiver_locations),
            data=data,
            dbdt_native=dbdt.copy(),
            dbdt_impulse=dbdt.copy(),
            primary_dbdt_reference="native_b",
            empymod_version="2.6.0",
            audit={
                "empymod_version": "2.6.0",
                "primary_dbdt_reference": "native_b",
                "native_b_available": True,
                "requested": True,
                "tolerance": 1.0e-12,
                "floor_fraction": 0.01,
                "performed": True,
                "passed": True,
            },
        )

    result = run_empymod_magnetic6_waveform_reference(
        _survey([0.1, 1.0]),
        waveform,
        quadrature_order=5,
        reference_runner=constant_runner,
    )

    np.testing.assert_allclose(result.data, 4.0)
    assert result.convolution["weight_sum"] == pytest.approx(1.0)
    assert result.convolution["quadrature_node_count"] == 15


def test_config_linear_ramp_builds_five_microsecond_waveform():
    config = {
        "source": {
            "current": 10.0,
            "waveform": {
                "type": "linear_ramp_off",
                "t_off": 5.0e-6,
                "current_initial": 10.0,
                "current_final": 0.0,
            },
        }
    }

    waveform = turnoff_waveform_from_config(config)

    assert waveform is not None
    np.testing.assert_allclose(waveform.times, [-5.0e-6, 0.0])
    np.testing.assert_allclose(waveform.values, [1.0, 0.0])


def test_config_step_off_requires_no_convolution():
    config = {
        "source": {
            "current": 10.0,
            "waveform": {"type": "step_off", "off_time": 0.0},
        }
    }

    assert turnoff_waveform_from_config(config) is None


def test_load_turnoff_csv_shifts_last_node_to_zero(tmp_path: Path):
    path = tmp_path / "waveform.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "current_scale"])
        writer.writeheader()
        writer.writerow({"time_s": 2.0e-6, "current_scale": 1.0})
        writer.writerow({"time_s": 5.0e-6, "current_scale": 0.4})
        writer.writerow({"time_s": 7.0e-6, "current_scale": 0.0})

    waveform = load_turnoff_csv(path)

    np.testing.assert_allclose(waveform.times, [-5.0e-6, -2.0e-6, 0.0])
    np.testing.assert_allclose(waveform.values, [1.0, 0.4, 0.0])


def test_cli_exposes_finite_waveform_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "case.yaml",
            "--numerical",
            "numerical.npz",
            "--depths",
            "0",
            "--resistivities",
            "1e8,100",
            "--output-dir",
            "out",
            "--ramp-off-time",
            "5e-6",
            "--waveform-quadrature-order",
            "10",
        ]
    )

    assert args.ramp_off_time == pytest.approx(5.0e-6)
    assert args.waveform_quadrature_order == 10
