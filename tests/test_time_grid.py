import numpy as np
import pytest

from atem3d.waveforms import (
    LinearRampOffWaveform,
    build_internal_time_grid,
    build_internal_time_grid_from_turnoff,
)


def test_time_grid_contains_turnoff_end_and_observation_times():
    waveform = LinearRampOffWaveform(
        current_initial=10.0,
        current_final=0.0,
        t_off=1.0e-5,
        min_steps_during_turnoff=10,
    )
    observation_times = np.array([1.0e-5, 1.0e-4, 1.0])

    grid = build_internal_time_grid(observation_times, waveform)

    assert grid[0] == pytest.approx(0.0)
    assert np.any(np.isclose(grid, waveform.t_off))
    for obs_time in observation_times:
        assert np.any(np.isclose(grid, waveform.t_off + obs_time))
    ramp_times = grid[(grid > 0.0) & (grid <= waveform.t_off)]
    assert ramp_times.size >= 10


def test_time_grid_from_turnoff_matches_waveform_grid():
    waveform = LinearRampOffWaveform(
        current_initial=10.0,
        current_final=0.0,
        t_off=1.0e-5,
        min_steps_during_turnoff=10,
    )
    observation_times = np.array([1.0e-5, 3.0e-5, 1.0])

    from_turnoff = build_internal_time_grid_from_turnoff(
        observation_times,
        turnoff_time=1.0e-5,
        turnoff_steps=10,
    )

    np.testing.assert_allclose(from_turnoff, build_internal_time_grid(observation_times, waveform))
    np.testing.assert_allclose(from_turnoff[:11], np.linspace(0.0, 1.0e-5, 11))
    np.testing.assert_allclose(from_turnoff[-3:], 1.0e-5 + observation_times)
