import numpy as np
import pytest

from atem3d.seepage_channel_validation import validate_result_payload


def test_empymod_is_background_only() -> None:
    payload = {
        "times": np.logspace(-5, -2, 31),
        "receiver_locations": np.asarray(
            [(0.0, y, -0.1) for y in (-20, -10, 0, 10, 20)]
        ),
        "components": np.array(["Ex", "dBzdt", "Hz"]),
        "values": np.ones((5, 31, 3)),
        "background_only_1d": True,
    }
    validate_result_payload("empymod", payload)
    payload["background_only_1d"] = False
    with pytest.raises(ValueError, match="background_only_1d"):
        validate_result_payload("empymod", payload)


def test_all_results_require_465_finite_values() -> None:
    payload = {
        "times": np.logspace(-5, -2, 31),
        "receiver_locations": np.asarray(
            [(0.0, y, -0.1) for y in (-20, -10, 0, 10, 20)]
        ),
        "components": np.array(["Ex", "dBzdt", "Hz"]),
        "values": np.ones((5, 31, 3)),
    }
    validate_result_payload("SimPEG", payload)
    payload["values"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_result_payload("SimPEG", payload)
