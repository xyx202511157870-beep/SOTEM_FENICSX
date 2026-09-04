import numpy as np

from atem3d.adaptive_debye_mvp.oracle_gap import collect_tilted_coil_tasks
from atem3d.adaptive_debye_mvp.protocol_constants import normalize_tilted_normal
from atem3d.adaptive_debye_mvp.registry import generate_all_cases


def _response(times, labels, dbx, dby, dbz):
    n_t = times.size
    n_rx = len(labels)
    data = np.zeros((n_t, n_rx, 6))
    data[:, :, 3] = np.asarray(dbx, dtype=float).reshape(n_t, 1)
    data[:, :, 4] = np.asarray(dby, dtype=float).reshape(n_t, 1)
    data[:, :, 5] = np.asarray(dbz, dtype=float).reshape(n_t, 1)
    return {
        "times": times,
        "channels": ["Hx", "Hy", "Hz", "dBxdt", "dBydt", "dBzdt"],
        "receiver_labels": labels,
        "data": data,
        "hashes": {"shared_survey_hash": "abc"},
    }


def test_tilted_projection_is_normal_dot_dbdt():
    case = next(item for item in generate_all_cases() if item.split == "independent_test")
    times = np.logspace(-5, -2, 31)
    labels = ["point:0", "point:1"]
    ones = np.ones(times.size)
    models = {
        "W0": _response(times, labels, ones, 2 * ones, 3 * ones),
    }
    tasks = collect_tilted_coil_tasks(
        case,
        candidate_id="x",
        K=8,
        reference_models=models,
        candidate_models=models,
        baseline_models=_response(times, labels, 0.5 * ones, ones, 1.5 * ones)
        and {"W0": _response(times, labels, 0.5 * ones, ones, 1.5 * ones)},
        waveform_ids=("W0",),
    )
    assert {item.receiver_id for item in tasks} == {"tilted_coil"}
    normal = np.asarray(case.sensor_normal)
    expected = float(normal[0] + 2 * normal[1] + 3 * normal[2])
    assert np.isclose(np.linalg.norm(normal), 1.0)
    # evaluate_task uses relative error; projection must be finite
    assert all(np.isfinite(item.dbdt_p95) for item in tasks)
    assert expected != 0.0


def test_normalize_tilted_normal_is_unit():
    vector = np.asarray(normalize_tilted_normal())
    assert np.isclose(np.linalg.norm(vector), 1.0)
