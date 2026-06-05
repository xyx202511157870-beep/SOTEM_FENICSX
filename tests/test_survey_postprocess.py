import numpy as np

from sotem_ip import FiniteWireSurvey, ip_percent_effect, relative_error


def test_default_survey_geometry():
    survey = FiniteWireSurvey()
    assert np.isclose(survey.source_length, 1000.0)
    assert np.isclose(survey.parallel_offset, 500.0)
    survey.validate(expected_length=1000.0, expected_offset=500.0)


def test_relative_error_uses_floor():
    err = relative_error([1.0, 0.0], [2.0, 0.0])
    assert np.isfinite(err).all()
    assert np.isclose(err[0], 0.5)


def test_ip_percent_effect():
    effect = ip_percent_effect([0.9, 2.2], [1.0, 2.0])
    assert np.allclose(effect, [-10.0, 10.0])

