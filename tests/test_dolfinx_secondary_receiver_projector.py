from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.constants import mu_0


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_secondary_projector_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_secondary_receiver_projector_bridge_uses_evaluate_receivers(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(0.0, -300.0, -0.1))
    seen = {}

    def electric_getter(state, Ep_new, time_value, dt):
        seen["electric_getter"] = (state, Ep_new.copy(), time_value, dt)
        return "E_secondary"

    def dbdt_getter(state, Ep_new, time_value, dt):
        seen["dbdt_getter"] = (state, Ep_new.copy(), time_value, dt)
        return "dBdt_secondary"

    def fake_evaluate_receivers(E, dbdt, msh, eval_config):
        seen["evaluate"] = (E, dbdt, msh, eval_config)
        return {"Ex": 1.0, "Ey": 2.0, "Hz": 3.0, "dBzdt": 4.0}

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate_receivers)
    projector = sp._make_secondary_receiver_projector_from_evaluate_receivers(
        electric_getter,
        dbdt_getter,
        msh="mesh",
        config=config,
    )

    values = projector("state", np.array([[0.5, 0.0, 0.0]]), 1.0e-5, 2.0e-6, ("Ex", "Ey", "dBzdt"))

    np.testing.assert_allclose(values, [[1.0, 2.0, 4.0]])
    assert seen["electric_getter"][0] == "state"
    np.testing.assert_allclose(seen["electric_getter"][1], [[0.5, 0.0, 0.0]])
    assert seen["electric_getter"][2] == 1.0e-5
    assert seen["electric_getter"][3] == 2.0e-6
    assert seen["evaluate"] == ("E_secondary", "dBdt_secondary", "mesh", config)


def test_secondary_receiver_projector_ampere_rate_only_overrides_dbdt(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(0.0, -300.0, -0.1),
        source_term_mode="primary_secondary",
        magnetic_dbdt_mode="ampere_rate",
    )
    diagnostics = {
        "secondary_H_receiver": np.asarray([0.0, 0.0, 3.5]),
        "secondary_dbdt_ampere_rate": np.asarray([0.0, 0.0, -8.0]),
    }

    def fake_evaluate_receivers(E, dbdt, msh, eval_config):
        return {"Ex": 1.0, "Ey": 2.0, "Hz": 3.0, "dBzdt": 4.0}

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate_receivers)
    projector = sp._make_secondary_receiver_projector_from_evaluate_receivers(
        lambda *_args: "E_secondary",
        lambda *_args: "dBdt_secondary",
        msh="mesh",
        config=config,
        diagnostics=diagnostics,
    )

    values = projector("state", np.array([[0.5, 0.0, 0.0]]), 1.0e-5, 2.0e-6, ("Ex", "Ey", "dBzdt"))

    np.testing.assert_allclose(values, [[1.0, 2.0, -8.0]])
    row = diagnostics["secondary_receiver_rows"][0]
    assert row["Ex"] == 1.0
    assert row["Ey"] == 2.0
    assert row["dBzdt_curl"] == 4.0
    assert row["dBzdt_ampere_rate"] == -8.0


def test_secondary_receiver_projector_uses_diagnostic_hz_when_requested(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(0.0, -300.0, -0.1),
        source_term_mode="primary_secondary",
        magnetic_receiver_mode="faraday_integrated",
    )
    diagnostics = {"secondary_H_receiver": np.asarray([0.0, 0.0, 3.5])}

    def fake_evaluate_receivers(E, dbdt, msh, eval_config):
        return {"Ex": 1.0, "Ey": 2.0, "dBzdt": 4.0}

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate_receivers)
    projector = sp._make_secondary_receiver_projector_from_evaluate_receivers(
        lambda *_args: "E_secondary",
        lambda *_args: "dBdt_secondary",
        msh="mesh",
        config=config,
        diagnostics=diagnostics,
    )

    values = projector("state", np.array([[0.5, 0.0, 0.0]]), 1.0e-5, 2.0e-6, ("Ex", "Ey", "Hz"))

    np.testing.assert_allclose(values, [[1.0, 2.0, 3.5]])
    row = diagnostics["secondary_receiver_rows"][0]
    assert row["Hz"] == 3.5


def test_secondary_receiver_projector_integrates_faraday_hz_when_no_h_diagnostic(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(0.0, -300.0, -0.1),
        source_term_mode="primary_secondary",
        magnetic_receiver_mode="faraday_integrated",
    )
    diagnostics = {}
    mu0 = 1.2566370614359173e-6

    def fake_evaluate_receivers(E, dbdt, msh, eval_config):
        return {"Ex": 1.0, "Ey": 2.0, "dBzdt": 2.0 * mu0}

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate_receivers)
    projector = sp._make_secondary_receiver_projector_from_evaluate_receivers(
        lambda *_args: "E_secondary",
        lambda *_args: "dBdt_secondary",
        msh="mesh",
        config=config,
        diagnostics=diagnostics,
    )

    first = projector("state", np.array([[0.5, 0.0, 0.0]]), 0.5, 0.5, ("Hz",))
    second = projector("state", np.array([[0.5, 0.0, 0.0]]), 1.0, 0.5, ("Hz",))

    np.testing.assert_allclose(first, [[1.0]])
    np.testing.assert_allclose(second, [[2.0]])
    assert diagnostics["secondary_receiver_rows"][0]["Hz"] == 1.0
    assert diagnostics["secondary_receiver_rows"][1]["Hz"] == 2.0


def test_secondary_receiver_projector_records_dbdt_candidate_diagnostics(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(receiver=(0.0, -300.0, -0.1))
    diagnostics = {}

    def fake_evaluate_receivers(E, dbdt, msh, eval_config):
        return {
            "Ex": 1.0,
            "Ey": 2.0,
            "dBzdt": 4.0,
            "dBzdt_candidate_min": -1.0,
            "dBzdt_candidate_median": 3.0,
            "dBzdt_candidate_mean": 5.0,
            "dBzdt_candidate_max": 9.0,
            "dBzdt_candidate_std": 2.5,
        }

    monkeypatch.setattr(sp, "evaluate_receivers", fake_evaluate_receivers)
    projector = sp._make_secondary_receiver_projector_from_evaluate_receivers(
        lambda *_args: "E_secondary",
        lambda *_args: "dBdt_secondary",
        msh="mesh",
        config=config,
        diagnostics=diagnostics,
    )

    projector("state", np.array([[0.5, 0.0, 0.0]]), 1.0e-5, 2.0e-6, ("Ex", "Ey", "dBzdt"))

    row = diagnostics["secondary_receiver_rows"][0]
    assert row["dBzdt_candidate_min"] == -1.0
    assert row["dBzdt_candidate_median"] == 3.0
    assert row["dBzdt_candidate_mean"] == 5.0
    assert row["dBzdt_candidate_max"] == 9.0
    assert row["dBzdt_candidate_std"] == 2.5


def test_evaluate_receivers_can_use_independent_dbdt_candidate_mode(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(0.0, -300.0, -0.1),
        receiver_evaluation_mode="shallowest",
        magnetic_dbdt_evaluation_mode="mean",
    )

    class FakeField:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def eval(self, points, cells):
            return self.values[: len(cells)]

    monkeypatch.setattr(sp, "_receiver_sampling_points", lambda _config: np.asarray([[0.0, -300.0, -0.1]]))
    monkeypatch.setattr(sp, "_find_cells_for_point", lambda _msh, _point: np.asarray([0, 1], dtype=np.int32))
    monkeypatch.setattr(
        sp,
        "_cell_centers",
        lambda _msh: np.asarray(
            [
                [0.0, -300.0, -2.0],
                [0.0, -300.0, -0.05],
            ],
            dtype=float,
        ),
    )

    electric = FakeField(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
        ]
    )
    dbdt = FakeField(
        [
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 8.0],
        ]
    )

    class FakeMesh:
        comm = None

    rec = sp.evaluate_receivers(electric, dbdt, msh=FakeMesh(), config=config)

    assert rec["Ex"] == 2.0
    assert rec["Ey"] == 20.0
    assert rec["dBzdt"] == 6.0


def test_h_receivers_use_independent_electric_and_dbdt_candidate_modes(monkeypatch):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        receiver=(0.0, -300.0, 0.0),
        receiver_evaluation_mode="shallowest",
        magnetic_dbdt_evaluation_mode="mean",
    )

    class FakeField:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def eval(self, points, cells):
            return self.values[np.asarray(cells, dtype=int)]

    monkeypatch.setattr(sp, "_find_cells_for_point", lambda _msh, _point: np.asarray([0, 1], dtype=np.int32))
    monkeypatch.setattr(
        sp,
        "_cell_centers",
        lambda _msh: np.asarray(
            [
                [0.0, -300.0, -2.0],
                [0.0, -300.0, -0.05],
            ],
            dtype=float,
        ),
    )

    electric = FakeField(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 0.0],
        ]
    )
    h_new = FakeField(
        [
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 8.0],
        ]
    )
    h_old = FakeField(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )

    rec = sp._evaluate_h_receivers(electric, h_new, h_old, 1.0, msh=object(), config=config)

    assert rec["Ex"] == 2.0
    assert rec["Ey"] == 20.0
    assert rec["Hz"] == 8.0
    np.testing.assert_allclose(rec["dBzdt"], 4.5 * mu_0)


def test_second_order_backward_derivative_supports_nonuniform_steps():
    sp = _load_pipeline_module()

    # f(t)=t^2 has exact derivative f'(3)=6.  The samples are nonuniform:
    # t_{n-2}=0, t_{n-1}=1, t_n=3.
    derivative = sp._second_order_backward_time_derivative(
        new=np.asarray([9.0]),
        old=np.asarray([1.0]),
        older=np.asarray([0.0]),
        dt=2.0,
        previous_dt=1.0,
    )

    np.testing.assert_allclose(derivative, np.asarray([6.0]))


def test_second_order_backward_derivative_falls_back_without_history():
    sp = _load_pipeline_module()

    derivative = sp._second_order_backward_time_derivative(
        new=np.asarray([5.0]),
        old=np.asarray([2.0]),
        older=None,
        dt=3.0,
        previous_dt=None,
    )

    np.testing.assert_allclose(derivative, np.asarray([1.0]))
