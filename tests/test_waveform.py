import numpy as np
import pytest

from atem3d.waveforms import LinearRampOffWaveform, StepOffWaveform, TabulatedWaveform


def test_linear_ramp_average_didt_matches_current_difference():
    waveform = LinearRampOffWaveform(
        current_initial=10.0,
        current_final=0.0,
        t_off=1.0e-5,
        min_steps_during_turnoff=10,
    )

    assert waveform.current(0.0) == pytest.approx(10.0)
    assert waveform.current(5.0e-6) == pytest.approx(5.0)
    assert waveform.current(1.0e-5) == pytest.approx(0.0)
    assert waveform.interval_average_didt(2.0e-6, 7.0e-6) == pytest.approx(-1.0e6)


def test_stepoff_average_didt_preserves_integral_over_interval_containing_off_time():
    waveform = StepOffWaveform(current_initial=10.0, t_off=1.0e-5)

    assert waveform.interval_average_didt(0.0, 2.0e-5) == pytest.approx(-5.0e5)
    assert waveform.current(0.5e-5) == pytest.approx(10.0)
    assert waveform.current(1.5e-5) == pytest.approx(0.0)


def test_tabulated_waveform_integral_equals_current_difference():
    waveform = TabulatedWaveform(
        times=np.array([0.0, 2.0e-6, 1.0e-5]),
        currents=np.array([10.0, 8.0, 0.0]),
        min_steps_during_turnoff=10,
    )

    assert waveform.current(1.0e-6) == pytest.approx(9.0)
    assert waveform.interval_average_didt(0.0, 1.0e-5) == pytest.approx(-1.0e6)

