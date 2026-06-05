import numpy as np
import pytest

from atem3d.waveforms import LinearRampOffWaveform, build_internal_time_grid


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
