import numpy as np

from atem3d.adaptive_debye_mvp.case_bridge import case_waveform
from atem3d.adaptive_debye_mvp.protocol_constants import WAVEFORM_BY_ID


def test_w3_protocol_nodes_and_scales():
    proto = WAVEFORM_BY_ID["W3"]
    assert proto.times_s == (-40.0e-6, -25.0e-6, -10.0e-6, 0.0)
    assert proto.current_scales == (1.0, 0.80, 0.25, 0.0)
    spec = case_waveform("W3")
    assert spec.kind == "tabulated"
    times = spec.config["source"]["waveform"]["times"]
    values = spec.config["source"]["waveform"]["values"]
    assert np.allclose(times, proto.times_s)
    assert np.allclose(values, proto.current_scales)


def test_run_layered_test_refuses_without_l1(tmp_path):
    from atem3d.adaptive_debye_mvp.io import write_json, read_json

    generated = tmp_path / "generated" / "receiver_adaptive_debye_mvp"
    generated.mkdir(parents=True)
    write_json(generated / "FLOW3_STATUS.json", {"status": "L1_POINTS_RANKED"})
    status = read_json(generated / "FLOW3_STATUS.json")
    assert status["status"] != "L1_FROZEN"
