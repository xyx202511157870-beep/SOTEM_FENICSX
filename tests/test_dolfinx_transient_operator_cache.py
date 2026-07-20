from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_transient_operator_cache_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Destroyable:
    def __init__(self, label: str, events: list[str]):
        self.label = label
        self.events = events
        self.destroy_count = 0

    def destroy(self):
        self.destroy_count += 1
        self.events.append(f"destroy:{self.label}")

    def assemble(self):
        pass


def _signature(sp, **overrides):
    values = {
        "dt": 2.5e-6,
        "time_method": "theta",
        "theta": 1.0,
        "bdf2_coefficients": None,
        "outer_boundary_mode": "pec",
        "robin_admittance": None,
        "divergence_control_applied": False,
        "divergence_control_weight": None,
        "debye_metadata": {"enabled": False, "reason": "no_debye_terms"},
    }
    values.update(overrides)
    return sp._transient_operator_signature(**values)


def _builders(events, generation: int):
    A = _Destroyable(f"A{generation}", events)
    M_eff = _Destroyable(f"M{generation}", events)
    ksp = _Destroyable(f"ksp{generation}", events)
    gradient = _Destroyable(f"G{generation}", events)

    def build_matrix():
        events.append(f"build:A{generation}")
        return A, M_eff, True

    def build_solver(matrix):
        assert matrix is A
        events.append(f"configure:ksp{generation}")
        return {"ksp": ksp, "G": gradient, "edge_constants": (object(), object(), object())}

    return (A, M_eff, ksp, gradient), build_matrix, build_solver


def test_same_operator_signature_reuses_matrix_and_ams_setup_once():
    sp = _load_pipeline_module()
    events: list[str] = []
    objects, build_matrix, build_solver = _builders(events, 1)
    cache = sp._TransientOperatorCache()

    first = cache.acquire(_signature(sp), build_matrix=build_matrix, build_solver=build_solver)
    second = cache.acquire(_signature(sp), build_matrix=build_matrix, build_solver=build_solver)

    assert first[:3] == second[:3]
    assert events == ["build:A1", "configure:ksp1"]
    assert first[3] is False
    assert second[3] is True
    cache.close()
    assert events == [
        "build:A1",
        "configure:ksp1",
        "destroy:ksp1",
        "destroy:G1",
        "destroy:A1",
        "destroy:M1",
    ]
    assert all(obj.destroy_count == 1 for obj in objects)


def test_four_linspace_substeps_report_one_build_then_three_reuses():
    sp = _load_pipeline_module()
    events: list[str] = []
    _objects, build_matrix, build_solver = _builders(events, 1)
    step_times = np.linspace(0.0, 1.0e-5, 5)[1:]
    dts = np.diff(np.r_[0.0, step_times])
    reused_flags = []

    with sp._TransientOperatorCache() as cache:
        for dt in dts:
            result = cache.acquire(
                _signature(sp, dt=float(dt)),
                build_matrix=build_matrix,
                build_solver=build_solver,
            )
            reused_flags.append(result[3])

    assert reused_flags == [False, True, True, True]
    assert events.count("configure:ksp1") == 1


def test_changed_signature_destroys_old_solver_and_matrices_before_rebuild():
    sp = _load_pipeline_module()
    events: list[str] = []
    _objects1, build_matrix1, build_solver1 = _builders(events, 1)
    _objects2, build_matrix2, build_solver2 = _builders(events, 2)
    cache = sp._TransientOperatorCache()

    cache.acquire(_signature(sp), build_matrix=build_matrix1, build_solver=build_solver1)
    cache.acquire(
        _signature(sp, dt=5.0e-6),
        build_matrix=build_matrix2,
        build_solver=build_solver2,
    )

    assert events == [
        "build:A1",
        "configure:ksp1",
        "destroy:ksp1",
        "destroy:G1",
        "destroy:A1",
        "destroy:M1",
        "build:A2",
        "configure:ksp2",
    ]
    cache.close()


def test_configure_ams_failure_destroys_ksp_before_gradient(monkeypatch):
    sp = _load_pipeline_module()
    events: list[str] = []
    gradient = _Destroyable("G", events)

    class FakePC:
        def setType(self, _value):
            pass

        def setHYPREType(self, _value):
            pass

        def setHYPREDiscreteGradient(self, _value):
            pass

    class FakeKSP(_Destroyable):
        def __init__(self):
            super().__init__("ksp", events)
            self.pc = FakePC()

        def create(self, _comm):
            return self

        def setOperators(self, _A):
            pass

        def setType(self, _value):
            pass

        def setTolerances(self, **_kwargs):
            pass

        def getPC(self):
            return self.pc

    fake_ksp = FakeKSP()
    petsc4py = ModuleType("petsc4py")
    petsc4py.PETSc = SimpleNamespace(KSP=lambda: fake_ksp)
    fem = ModuleType("dolfinx.fem")

    def fail_function(_space):
        raise RuntimeError("edge constant setup failed")

    fem.Function = fail_function
    fem_petsc = ModuleType("dolfinx.fem.petsc")
    fem_petsc.discrete_gradient = lambda _S, _V: gradient
    fem.petsc = fem_petsc
    dolfinx = ModuleType("dolfinx")
    dolfinx.fem = fem
    monkeypatch.setitem(sys.modules, "petsc4py", petsc4py)
    monkeypatch.setitem(sys.modules, "dolfinx", dolfinx)
    monkeypatch.setitem(sys.modules, "dolfinx.fem", fem)
    monkeypatch.setitem(sys.modules, "dolfinx.fem.petsc", fem_petsc)

    with pytest.raises(SystemExit, match="edge constant setup failed"):
        sp.configure_ams_solver(
            SimpleNamespace(comm=object()),
            {"V": object(), "S": object()},
            SimpleNamespace(ksp_type="cg", rtol=1e-8, atol=1e-12, max_it=10),
        )

    assert events == ["destroy:ksp", "destroy:G"]


@pytest.mark.parametrize("builder", ["combined", "debye"])
def test_matrix_build_helpers_destroy_partial_copy_on_failure(monkeypatch, builder):
    sp = _load_pipeline_module()
    events: list[str] = []

    class FailingMatrix(_Destroyable):
        def scale(self, _value):
            pass

        def axpy(self, *_args, **_kwargs):
            raise RuntimeError("matrix combination failed")

    partial = FailingMatrix("partial", events)
    source_matrix = SimpleNamespace(copy=lambda: partial)
    petsc4py = ModuleType("petsc4py")
    petsc4py.PETSc = SimpleNamespace(
        Mat=SimpleNamespace(
            Structure=SimpleNamespace(SAME_NONZERO_PATTERN=object()),
        )
    )
    monkeypatch.setitem(sys.modules, "petsc4py", petsc4py)

    with pytest.raises(RuntimeError, match="matrix combination failed"):
        if builder == "combined":
            sp._copy_and_combine_matrix(source_matrix, object(), 2.0)
        else:
            sp._matrix_for_effective_conductivity(
                {"M_inf": source_matrix, "M_debye": [object()]},
                {"terms": [SimpleNamespace(tau=1.0, delta_sigma=0.1)]},
                0.1,
            )

    assert partial.destroy_count == 1


def test_context_manager_cleans_cached_objects_when_step_raises():
    sp = _load_pipeline_module()
    events: list[str] = []
    objects, build_matrix, build_solver = _builders(events, 1)

    with pytest.raises(RuntimeError, match="step failed"):
        with sp._TransientOperatorCache() as cache:
            cache.acquire(_signature(sp), build_matrix=build_matrix, build_solver=build_solver)
            raise RuntimeError("step failed")

    assert [obj.destroy_count for obj in objects] == [1, 1, 1, 1]
    assert events[-4:] == ["destroy:ksp1", "destroy:G1", "destroy:A1", "destroy:M1"]


def test_public_forward_wrapper_cleans_cache_when_implementation_raises(monkeypatch):
    sp = _load_pipeline_module()
    events: list[str] = []
    objects, build_matrix, build_solver = _builders(events, 1)

    def fail_forward(*_args, operator_cache, **_kwargs):
        operator_cache.acquire(
            _signature(sp),
            build_matrix=build_matrix,
            build_solver=build_solver,
        )
        raise RuntimeError("forward failed")

    monkeypatch.setattr(sp, "_run_fetd_forward_impl", fail_forward)
    with pytest.raises(RuntimeError, match="forward failed"):
        sp.run_fetd_forward(None, None, None, None, None, None, None)

    assert [obj.destroy_count for obj in objects] == [1, 1, 1, 1]
    assert events[-4:] == ["destroy:ksp1", "destroy:G1", "destroy:A1", "destroy:M1"]


def test_failed_solver_setup_destroys_new_matrices_and_leaves_cache_empty():
    sp = _load_pipeline_module()
    events: list[str] = []
    A = _Destroyable("A1", events)
    M_eff = _Destroyable("M1", events)

    def build_matrix():
        events.append("build:A1")
        return A, M_eff, True

    def fail_solver(_matrix):
        events.append("configure:failed")
        raise RuntimeError("AMS setup failed")

    cache = sp._TransientOperatorCache()
    with pytest.raises(RuntimeError, match="AMS setup failed"):
        cache.acquire(_signature(sp), build_matrix=build_matrix, build_solver=fail_solver)

    assert events == ["build:A1", "configure:failed", "destroy:A1", "destroy:M1"]
    assert cache.signature is None
    cache.close()
    assert A.destroy_count == 1
    assert M_eff.destroy_count == 1


@pytest.mark.parametrize(
    "override",
    [
        {"dt": 5.0e-6},
        {"time_method": "bdf2", "bdf2_coefficients": {"lhs": 6.0e5, "old": 8.0e5, "older": -2.0e5}},
        {"theta": 0.5},
        {"outer_boundary_mode": "robin", "robin_admittance": 12.0},
        {"divergence_control_applied": True, "divergence_control_weight": 1.0e-3},
        {
            "debye_metadata": {
                "enabled": True,
                "sigma_eff": 0.009,
                "delta_sigma": [0.001],
                "tau": [0.01],
                "alpha": [0.99975],
                "beta": [0.00025],
            }
        },
    ],
)
def test_signature_changes_for_every_operator_condition(override):
    sp = _load_pipeline_module()

    assert _signature(sp, **override) != _signature(sp)


def test_signature_collapses_only_roundoff_level_linspace_dt_noise():
    sp = _load_pipeline_module()
    step_times = np.linspace(0.0, 1.0e-5, 5)[1:]
    dts = np.diff(np.r_[0.0, step_times])

    signatures = {
        _signature(sp, dt=float(dt), theta=1.0)
        for dt in dts
    }

    assert len(signatures) == 1
    assert _signature(sp, dt=2.5e-6 * (1.0 + 1.0e-10)) not in signatures


def test_same_dt_does_not_reuse_distinct_robin_divergence_bdf2_or_debye_operators():
    sp = _load_pipeline_module()
    baseline = _signature(sp)

    assert _signature(
        sp,
        outer_boundary_mode="robin",
        robin_admittance=10.0,
    ) != _signature(
        sp,
        outer_boundary_mode="robin",
        robin_admittance=11.0,
    )
    assert _signature(
        sp,
        divergence_control_applied=True,
        divergence_control_weight=1e-3,
    ) != _signature(
        sp,
        divergence_control_applied=True,
        divergence_control_weight=2e-3,
    )
    assert _signature(
        sp,
        divergence_control_applied=True,
        divergence_control_weight=1e-3,
    ) != baseline

    current_dt = 2.5e-6
    coeffs_equal_previous_dt = sp._bdf2_step_coefficients(current_dt, current_dt)
    coeffs_longer_previous_dt = sp._bdf2_step_coefficients(current_dt, 5.0e-6)
    assert _signature(
        sp,
        time_method="bdf2",
        bdf2_coefficients=coeffs_equal_previous_dt,
    ) != _signature(
        sp,
        time_method="bdf2",
        bdf2_coefficients=coeffs_longer_previous_dt,
    )

    debye_a = {
        "enabled": True,
        "sigma_eff": 0.009,
        "delta_sigma": [0.001],
        "tau": [0.01],
        "alpha": [0.99975],
        "beta": [0.00025],
    }
    debye_b = dict(debye_a, sigma_eff=0.0085, beta=[0.0003])
    assert _signature(sp, debye_metadata=debye_a) != _signature(sp, debye_metadata=debye_b)
