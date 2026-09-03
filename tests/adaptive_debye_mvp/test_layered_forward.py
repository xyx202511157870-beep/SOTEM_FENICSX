import numpy as np
import pytest

from atem3d.adaptive_debye_mvp.layered_forward import (
    average_receiver_channels,
    delayed_time_union,
    noip_resistivities,
    pelton_layers,
    run_case_models,
    waveform_from_id,
)
from atem3d.adaptive_debye_mvp.protocol_constants import observation_times
from atem3d.adaptive_debye_mvp.registry import generate_split
from atem3d.empymod_waveform import PiecewiseLinearTurnOff


def _tiny_case():
    return generate_split("pilot_gap")[0]


def test_waveforms_share_shutoff_origin():
    times = observation_times()
    assert times[0] == pytest.approx(1.0e-5)
    assert times[-1] == pytest.approx(1.0e-2)
    assert waveform_from_id("W0") is None
    w1 = waveform_from_id("W1")
    w2 = waveform_from_id("W2")
    w3 = waveform_from_id("W3")
    assert isinstance(w1, PiecewiseLinearTurnOff)
    assert w1.times[-1] == pytest.approx(0.0)
    assert w2.times[-1] == pytest.approx(0.0)
    assert w3.times[-1] == pytest.approx(0.0)
    assert w1.duration == pytest.approx(5.0e-6)
    assert w2.duration == pytest.approx(20.0e-6)
    union = delayed_time_union(times, ("W0", "W1", "W2"))
    assert union[0] > 0.0
    assert times[0] in set(np.round(union, 12)) or np.min(np.abs(union - times[0])) < 1.0e-15


def test_m_zero_ip_increment_is_numerical_zero():
    from atem3d.adaptive_debye_mvp.receiver_metrics import ReceiverCase, evaluate_case

    case = _tiny_case()
    times = observation_times()[::5]
    zero_ip = run_case_models(
        case,
        resistivities=pelton_layers(case, chargeability=0.0),
        model_id="test_m0",
        waveform_ids=("W0",),
        times=times,
        include_disks=False,
    )["W0"]
    repeat = run_case_models(
        case,
        resistivities=pelton_layers(case, chargeability=0.0),
        model_id="test_m0_repeat",
        waveform_ids=("W0",),
        times=times,
        include_disks=False,
    )["W0"]
    np.testing.assert_allclose(zero_ip.data, repeat.data, rtol=0.0, atol=0.0)
    location = average_receiver_channels(zero_ip, case.receivers[0], kind="point")
    metrics = evaluate_case(
        ReceiverCase(
            case_id="m0",
            times=times,
            reference=location,
            candidate=location,
            reference_no_ip=location,
        )
    )
    assert metrics.case_ip_increment_nrmse == 0.0 or not np.isfinite(metrics.case_ip_increment_nrmse) or metrics.case_ip_increment_nrmse < 1.0e-12


def test_point_and_tiny_disk_converge():
    case = _tiny_case()
    times = observation_times()[::6]
    response = run_case_models(
        case,
        resistivities=noip_resistivities(case),
        model_id="test_disk",
        waveform_ids=("W0",),
        times=times,
        include_disks=True,
        disk_rule="square4",
    )["W0"]
    location = case.receivers[0]
    point = average_receiver_channels(response, location, kind="point")
    disk = average_receiver_channels(response, location, kind="disk", radius=0.25)
    for name in ("Hz", "dBzdt"):
        scale = max(float(np.max(np.abs(point[name]))), 1.0e-16)
        assert float(np.max(np.abs(disk[name] - point[name]))) / scale < 0.25


def test_exact_and_candidate_share_survey_geometry():
    from atem3d.adaptive_debye_mvp.layered_forward import build_survey, unique_evaluation_locations

    case = _tiny_case()
    survey = build_survey(case)
    assert survey.source_start == case.source_start
    assert survey.source_end == case.source_end
    assert survey.receiver_locations == case.receivers
    locations = unique_evaluation_locations(case, include_disks=True, disk_rule="square4")
    assert case.receivers[0] in locations
    assert len(locations) > len(case.receivers)
