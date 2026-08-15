from __future__ import annotations

import csv
from pathlib import Path
import re

import numpy as np
import pytest
from scipy.constants import mu_0

from atem3d.empymod_compare import EmpymodSurvey
from atem3d.empymod_magnetic6 import (
    MAGNETIC6_COMPONENTS,
    MAGNETIC6_UNITS,
    MagneticSixNumericalData,
    build_magnetic6_survey_from_config,
    compare_magnetic6,
    load_magnetic6_numerical,
    receiver_locations_from_config,
    run_empymod_magnetic6_reference,
)


class ConsistentFakeEmpymod:
    __version__ = "2.6.0"

    def __init__(self):
        self.calls = []

    def bipole(self, **kwargs):
        self.calls.append(kwargs)
        rec = kwargs["rec"]
        azimuth = float(rec[3])
        dip = float(rec[4])
        if abs(abs(dip) - 90.0) < 1.0e-12:
            axis = 3.0
        elif abs(azimuth - 90.0) < 1.0e-12:
            axis = 2.0
        else:
            axis = 1.0
        count = len(kwargs["freqtime"])
        if kwargs["mrec"] == "b":
            return np.full(count, axis * 1.0e-9)
        if kwargs["mrec"] is True and kwargs["signal"] == 0:
            return np.full(count, -axis * 1.0e-9 / mu_0)
        if kwargs["mrec"] is True and kwargs["signal"] == -1:
            return np.full(count, axis * 10.0)
        raise AssertionError(kwargs)


class LegacyFakeEmpymod(ConsistentFakeEmpymod):
    __version__ = "2.5.4"

    def bipole(self, **kwargs):
        if kwargs["mrec"] == "b":
            raise AssertionError("legacy backend must not receive mrec='b'")
        return super().bipole(**kwargs)


def _survey(coordinate_system="depth_down"):
    z = 0.2 if coordinate_system == "depth_down" else -0.2
    source_z = 0.1 if coordinate_system == "depth_down" else -0.1
    return EmpymodSurvey(
        source_start=(-10.0, -5.0, source_z),
        source_end=(10.0, 5.0, source_z),
        receiver_locations=[(7.0, 13.0, z)],
        components=MAGNETIC6_COMPONENTS,
        times=np.array([1.0e-4, 1.0e-3]),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        strength=2.0,
        signal=-1,
        coordinate_system=coordinate_system,
    )


def _version_tuple(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    assert match is not None
    return tuple(int(value) for value in match.groups())


def test_magnetic6_reference_uses_native_b_and_impulse_audit():
    backend = ConsistentFakeEmpymod()

    result = run_empymod_magnetic6_reference(
        _survey(),
        backend=backend,
        srcpts=7,
        recpts=1,
        dbdt_reference="auto",
        audit_impulse=True,
        require_audit_pass=True,
    )

    assert result.data.shape == (2, 1, 6)
    assert result.components == MAGNETIC6_COMPONENTS
    assert result.units == MAGNETIC6_UNITS
    assert result.primary_dbdt_reference == "native_b"
    assert result.empymod_version == "2.6.0"
    np.testing.assert_allclose(result.data[0, 0, :3], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        result.data[0, 0, 3:], [1.0e-9, 2.0e-9, 3.0e-9]
    )
    assert result.audit["performed"] is True
    assert result.audit["passed"] is True
    assert len(backend.calls) == 9
    assert all(call["srcpts"] == 7 for call in backend.calls)
    assert sum(
        call["mrec"] is True and call["signal"] == -1
        for call in backend.calls
    ) == 3
    assert sum(
        call["mrec"] == "b" and call["signal"] == -1
        for call in backend.calls
    ) == 3
    assert sum(
        call["mrec"] is True and call["signal"] == 0
        for call in backend.calls
    ) == 3


def test_auto_falls_back_to_impulse_h_for_empymod_254():
    backend = LegacyFakeEmpymod()

    result = run_empymod_magnetic6_reference(
        _survey(),
        backend=backend,
        dbdt_reference="auto",
        audit_impulse=True,
    )

    assert result.primary_dbdt_reference == "impulse_h"
    assert result.dbdt_native is None
    assert result.dbdt_impulse is not None
    assert result.audit["performed"] is False
    assert result.audit["passed"] is True
    assert result.audit["native_b_available"] is False
    np.testing.assert_allclose(result.data[0, 0, :3], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        result.data[0, 0, 3:], [1.0e-9, 2.0e-9, 3.0e-9]
    )
    assert len(backend.calls) == 6
    assert all(call["mrec"] is True for call in backend.calls)


def test_magnetic6_reference_applies_z_up_axial_vector_signs():
    result = run_empymod_magnetic6_reference(
        _survey("z_up"),
        backend=ConsistentFakeEmpymod(),
        dbdt_reference="auto",
        audit_impulse=True,
    )

    np.testing.assert_allclose(
        result.data[0, 0, :3], [-10.0, -20.0, 30.0]
    )
    np.testing.assert_allclose(
        result.data[0, 0, 3:], [-1.0e-9, -2.0e-9, 3.0e-9]
    )
    assert result.audit["passed"] is True


def test_native_b_requires_empymod_26_when_version_is_known():
    with pytest.raises(RuntimeError, match="empymod >= 2.6"):
        run_empymod_magnetic6_reference(
            _survey(),
            backend=LegacyFakeEmpymod(),
            dbdt_reference="native_b",
        )


def test_required_crosscheck_fails_when_native_route_is_unavailable():
    with pytest.raises(
        RuntimeError,
        match="cross-check was required but is unavailable",
    ):
        run_empymod_magnetic6_reference(
            _survey(),
            backend=LegacyFakeEmpymod(),
            dbdt_reference="auto",
            audit_impulse=True,
            require_audit_pass=True,
        )


def test_build_magnetic6_survey_uses_each_unique_location_once():
    config = {
        "coordinate_system": "z_up",
        "source": {
            "start": [-10.0, 0.0, -0.1],
            "end": [10.0, 0.0, -0.1],
            "current": 3.0,
        },
        "receivers": [
            {"location": [0.0, 10.0, -0.2], "component": "Hx"},
            {"location": [0.0, 10.0, -0.2], "component": "Hz"},
            {"location": [5.0, 10.0, -0.2], "component": "dBzdt"},
        ],
    }

    survey = build_magnetic6_survey_from_config(
        config,
        times=[1.0e-4],
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
    )

    assert survey.receiver_locations == (
        (0.0, 10.0, -0.2),
        (5.0, 10.0, -0.2),
    )
    assert len(survey.receiver_components) == 12
    assert [
        component for _location, component in survey.receiver_components[:6]
    ] == list(MAGNETIC6_COMPONENTS)


def test_receiver_line_coordinates_broadcast():
    config = {
        "receiver_line": {
            "x": [-1.0, 0.0, 1.0],
            "y": 2.0,
            "z": -0.5,
        }
    }
    assert receiver_locations_from_config(config) == (
        (-1.0, 2.0, -0.5),
        (0.0, 2.0, -0.5),
        (1.0, 2.0, -0.5),
    )


def test_npz_loader_reorders_components(tmp_path: Path):
    components = np.array(
        ["Hz", "Hx", "dBzdt", "Hy", "dBxdt", "dBydt"]
    )
    data = np.arange(12.0).reshape(2, 1, 6)
    path = tmp_path / "numerical.npz"
    np.savez(
        path,
        times=np.array([1.0, 2.0]),
        data=data,
        components=components,
    )

    loaded = load_magnetic6_numerical(path)

    expected_indices = [1, 3, 0, 4, 5, 2]
    np.testing.assert_allclose(loaded.data, data[..., expected_indices])


def test_csv_loader_and_comparison(tmp_path: Path):
    path = tmp_path / "numerical.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["time_obs", *MAGNETIC6_COMPONENTS],
        )
        writer.writeheader()
        for time in (1.0e-4, 1.0e-3):
            writer.writerow(
                {
                    "time_obs": time,
                    **{
                        component: index + 1.0
                        for index, component in enumerate(MAGNETIC6_COMPONENTS)
                    },
                }
            )

    numerical = load_magnetic6_numerical(path)
    reference = run_empymod_magnetic6_reference(
        _survey(),
        backend=ConsistentFakeEmpymod(),
        audit_impulse=True,
    )
    exact = MagneticSixNumericalData(
        times=reference.times,
        data=reference.data.copy(),
    )
    comparison = compare_magnetic6(exact, reference, tolerance=1.0e-12)

    assert numerical.data.shape == (2, 1, 6)
    assert comparison["passed"] is True
    assert comparison["global_max_floor_relative_error"] == 0.0
    assert comparison["primary_dbdt_reference"] == "native_b"


def test_real_empymod_magnetic6_reference_is_finite():
    empymod = pytest.importorskip("empymod")
    survey = EmpymodSurvey(
        source_start=(-20.0, -7.0, 0.1),
        source_end=(20.0, 7.0, 0.1),
        receiver_locations=[(13.0, 31.0, 0.2)],
        components=MAGNETIC6_COMPONENTS,
        times=np.logspace(-4, -2, 4),
        depths=[0.0],
        resistivities=[1.0e8, 100.0],
        strength=1.0,
        signal=-1,
        coordinate_system="depth_down",
    )

    result = run_empymod_magnetic6_reference(
        survey,
        backend=empymod,
        srcpts=3,
        recpts=1,
        dbdt_reference="auto",
        audit_impulse=True,
        audit_tolerance=0.05,
        audit_floor_fraction=0.01,
    )

    assert np.all(np.isfinite(result.data))
    version = _version_tuple(empymod.__version__)
    if version >= (2, 6):
        assert result.primary_dbdt_reference == "native_b"
        assert result.dbdt_native is not None
        assert result.audit["performed"] is True
        assert result.audit["passed"] is True
    else:
        assert result.primary_dbdt_reference == "impulse_h"
        assert result.dbdt_native is None
        assert result.dbdt_impulse is not None
        assert result.audit["performed"] is False
        assert result.audit["passed"] is True
