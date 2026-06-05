import numpy as np
import pytest

from atem3d.forward_operator import ForwardOperator, ForwardRequest


def test_forward_operator_calls_runner_with_model_survey_waveform_and_times():
    seen = {}

    def runner(request):
        seen["request"] = request
        return np.array([[1.0, 2.0], [3.0, 4.0]])

    model = {"sigma": 0.01}
    survey = {"receivers": ["rx1", "rx2"]}
    waveform = {"type": "linear_ramp_off"}
    times = [1.0e-5, 2.0e-5]
    operator = ForwardOperator(runner=runner)

    predicted_data = operator.forward(model, survey, waveform, times)

    assert isinstance(seen["request"], ForwardRequest)
    assert seen["request"].model is model
    assert seen["request"].survey is survey
    assert seen["request"].waveform is waveform
    np.testing.assert_allclose(seen["request"].times, times)
    np.testing.assert_allclose(predicted_data, [[1.0, 2.0], [3.0, 4.0]])


def test_forward_operator_rejects_nonincreasing_times_before_calling_runner():
    called = False

    def runner(_request):
        nonlocal called
        called = True
        return np.zeros((1, 1))

    operator = ForwardOperator(runner=runner)

    with pytest.raises(ValueError, match="strictly increasing"):
        operator.forward(model={}, survey={}, waveform={}, times=[1.0, 1.0])

    assert called is False


def test_forward_operator_rejects_non_2d_predicted_data():
    operator = ForwardOperator(runner=lambda _request: np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="predicted_data"):
        operator.forward(model={}, survey={}, waveform={}, times=[1.0])
