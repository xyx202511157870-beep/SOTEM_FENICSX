from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_e_refresh_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_e_step_solver_refreshes_existing_ams_when_matrix_signature_changes(monkeypatch):
    sp = _load_pipeline_module()
    calls = []

    def fail_if_recreated(A, spaces, config):
        raise AssertionError("a changed dt must refresh, not recreate, the AMS context")

    def fake_refresh(context, matrix, config):
        calls.append((context, matrix, config))

    monkeypatch.setattr(sp, "configure_ams_solver", fail_if_recreated)
    monkeypatch.setattr(sp, "_refresh_ams_solver_context", fake_refresh)
    config = sp.PipelineConfig(formulation="e")
    spaces = {"V": object(), "S": object()}
    previous = {
        "ksp": object(),
        "G": object(),
        "edge_constants": (object(), object(), object()),
        "matrix_signature": (1.0e-9, False, 0.0),
    }

    context = sp._configure_e_step_solver(
        previous,
        "new-matrix",
        spaces,
        config,
        matrix_signature=(1.0e-7, False, 0.0),
    )

    assert context is previous
    assert context["matrix_signature"] == (1.0e-7, False, 0.0)
    assert calls == [(previous, "new-matrix", config)]


def test_destroy_solver_context_releases_petsc_resources(monkeypatch):
    sp = _load_pipeline_module()
    release_calls = []

    class Destroyable:
        def __init__(self):
            self.destroyed = False

        def destroy(self):
            self.destroyed = True

    ksp = Destroyable()
    gradient = Destroyable()
    context = {"ksp": ksp, "G": gradient, "edge_constants": (object(),)}
    monkeypatch.setattr(sp, "_release_native_solver_memory", lambda: release_calls.append(True))

    sp._destroy_solver_context(context)

    assert ksp.destroyed is True
    assert gradient.destroyed is True
    assert context == {}
    assert release_calls == [True]


def test_e_step_solver_reuses_ams_for_roundoff_equivalent_signature(monkeypatch):
    sp = _load_pipeline_module()

    class FakePC:
        def __init__(self):
            self.reuse = []

        def setReusePreconditioner(self, value):
            self.reuse.append(value)

    class FakeKSP:
        def __init__(self):
            self.operators = []
            self.pc = FakePC()

        def setOperators(self, matrix):
            self.operators.append(matrix)

        def getPC(self):
            return self.pc

    def fail_if_reconfigured(A, spaces, config):
        raise AssertionError("roundoff-equivalent dt must reuse the existing AMS context")

    monkeypatch.setattr(sp, "configure_ams_solver", fail_if_reconfigured)
    ksp = FakeKSP()
    previous = {"ksp": ksp, "matrix_signature": (1.0e-7, False, 0.0)}

    context = sp._configure_e_step_solver(
        previous,
        "same-matrix-values",
        {"V": object(), "S": object()},
        sp.PipelineConfig(formulation="e"),
        matrix_signature=(1.0e-7 + 1.0e-20, False, 0.0),
    )

    assert context is previous
    assert ksp.operators == ["same-matrix-values"]


def test_e_step_solver_refreshes_preconditioner_for_moderate_dt_change(monkeypatch):
    sp = _load_pipeline_module()
    refresh_calls = []

    class FakePC:
        def __init__(self):
            self.reuse = []

        def setReusePreconditioner(self, value):
            self.reuse.append(value)

    class FakeKSP:
        def __init__(self):
            self.operators = []
            self.pc = FakePC()

        def setOperators(self, matrix):
            self.operators.append(matrix)

        def getPC(self):
            return self.pc

    def fake_refresh(context, matrix, config):
        refresh_calls.append((context, matrix, config))

    monkeypatch.setattr(sp, "_refresh_ams_solver_context", fake_refresh)
    ksp = FakeKSP()
    config = sp.PipelineConfig(formulation="e")
    previous = {
        "ksp": ksp,
        "matrix_signature": (1.0e-7, False, 0.0),
        "preconditioner_dt": 1.0e-7,
    }

    context = sp._configure_e_step_solver(
        previous,
        "moderately-changed-matrix",
        {"V": object(), "S": object()},
        config,
        matrix_signature=(1.2e-7, False, 0.0),
    )

    assert context is previous
    assert context["matrix_signature"] == (1.2e-7, False, 0.0)
    assert context["preconditioner_dt"] == 1.2e-7
    assert refresh_calls == [(previous, "moderately-changed-matrix", config)]


def test_refresh_replaces_ksp_to_drop_hypre_state(monkeypatch):
    sp = _load_pipeline_module()
    events = []

    class FakePC:
        def setType(self, value):
            events.append(("type", value))

        def setHYPREType(self, value):
            events.append(("hypre", value))

        def setHYPREDiscreteGradient(self, value):
            events.append(("gradient", value))

        def setHYPRESetEdgeConstantVectors(self, *values):
            events.append(("edge_constants", values))

        def setReusePreconditioner(self, value):
            events.append(("reuse", value))

        def setUp(self):
            events.append(("setup", None))

    class FakeKSP:
        def __init__(self, name):
            self.name = name
            self.pc = FakePC()

        def reset(self):
            events.append((self.name, "reset"))

        def destroy(self):
            events.append((self.name, "destroy"))

        def setOperators(self, value):
            events.append(("operators", value))

        def setType(self, value):
            events.append(("ksp_type", value))

        def setTolerances(self, **values):
            events.append(("tolerances", values))

        def getPC(self):
            return self.pc

    replacement = FakeKSP("replacement")

    class FakeKSPFactory:
        def create(self, comm):
            events.append(("create", comm))
            return replacement

    class FakeEdgeConstant:
        class X:
            petsc_vec = object()

        x = X()

    monkeypatch.setattr(sp, "_release_native_solver_memory", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "petsc4py",
        SimpleNamespace(PETSc=SimpleNamespace(KSP=FakeKSPFactory)),
    )
    old_ksp = FakeKSP("old")
    context = {
        "ksp": old_ksp,
        "G": "gradient",
        "edge_constants": tuple(FakeEdgeConstant() for _ in range(3)),
    }
    matrix = SimpleNamespace(comm="world")

    sp._refresh_ams_solver_context(context, matrix, sp.PipelineConfig())

    assert ("old", "destroy") in events
    assert ("old", "reset") not in events
    assert ("create", "world") in events
    assert context["ksp"] is replacement
    assert events[-1] == ("reuse", True)


def test_e_forward_loop_uses_matrix_signature_refresh():
    sp = _load_pipeline_module()

    source = inspect.getsource(sp.run_fetd_forward)

    assert "_configure_e_step_solver(" in source


def test_e_forward_loop_releases_final_solver_context():
    sp = _load_pipeline_module()

    source = inspect.getsource(sp.run_fetd_forward)

    assert "_destroy_solver_context(solver_context)" in source
