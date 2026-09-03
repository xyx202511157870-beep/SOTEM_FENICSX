from __future__ import annotations

import numpy as np

from atem3d.adaptive_debye_mvp.case_bridge import (
    assert_shared_survey_hash,
    case_geometry,
    case_time_grid,
    case_waveform,
    debye_material,
    disk_receivers,
    exact_pelton_material,
    forward_response,
    nonpolarizable_material,
    point_receivers,
    polarizable_material,
)
from atem3d.adaptive_debye_mvp.layered_forward import (
    SMOKE_FAST_TRANSFORM,
    DebyeCandidateMaterial,
    ExactPeltonMaterial,
    NonPolarizableMaterial,
)
from atem3d.adaptive_debye_mvp.passive_fit import fit_pelton_passive_hard_dc
from atem3d.adaptive_debye_mvp.protocol_constants import COORDINATE_SYSTEM, SPECTRAL_FREQUENCIES
from atem3d.adaptive_debye_mvp.registry import generate_split


def test_case_bridge_types_and_depth_down():
    case = generate_split("pilot_gap")[0]
    geometry = case_geometry(case)
    assert geometry.coordinate_system == COORDINATE_SYSTEM
    assert geometry.n_layers == len(case.resistivities)
    assert isinstance(exact_pelton_material(case), ExactPeltonMaterial)
    assert isinstance(nonpolarizable_material(case), NonPolarizableMaterial)
    points = point_receivers(case)
    disks = disk_receivers(case)
    assert len(points) == 2
    assert points[0].kind == "point"
    assert [item.kind for item in disks] == ["disk_average", "disk_average"]
    assert case_waveform("W0").kind == "ideal_step_off"
    assert case_waveform("W1").kind == "linear_ramp"
    assert case_waveform("W2").ramp_duration_s == 20.0e-6
    material = polarizable_material(case)
    sigma = material.complex_conductivity(SPECTRAL_FREQUENCIES)
    assert sigma.shape == SPECTRAL_FREQUENCIES.shape


def test_shared_survey_hash_identical_across_constitutive_models():
    case = generate_split("pilot_gap")[0]
    times = case_time_grid(np.logspace(-5, -2, 5))
    receivers = (point_receivers(case)[0],)
    geometry = case_geometry(case)
    waveform = case_waveform("W0")
    exact = forward_response(
        exact_pelton_material(case),
        geometry,
        waveform,
        receivers,
        times,
        SMOKE_FAST_TRANSFORM,
    )
    noip = forward_response(
        nonpolarizable_material(case),
        geometry,
        waveform,
        receivers,
        times,
        SMOKE_FAST_TRANSFORM,
    )
    pelton = polarizable_material(case)
    fit = fit_pelton_passive_hard_dc(
        pelton.rho0,
        pelton.chargeability,
        pelton.tau,
        pelton.c,
        SPECTRAL_FREQUENCIES,
        np.array([1.0e-4, 1.0e-3, 1.0e-2]),
    )
    debye = debye_material(case, fit, candidate_id="bridge_smoke")
    assert isinstance(debye, DebyeCandidateMaterial)
    candidate = forward_response(
        debye,
        geometry,
        waveform,
        receivers,
        times,
        SMOKE_FAST_TRANSFORM,
    )
    shared = assert_shared_survey_hash([exact, noip, candidate])
    assert shared
    assert exact["data"].shape == (5, 1, 6)
    delta = np.asarray(exact["data"]) - np.asarray(noip["data"])
    assert float(np.max(np.abs(delta))) > 0.0
