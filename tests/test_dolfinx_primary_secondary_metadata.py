from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from atem3d.materials.prony import DebyeTerm, PronyConductivity


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_primary_secondary_metadata",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_record_primary_secondary_step_equation_writes_noip_metadata():
    sp = _load_pipeline_module()
    diagnostics = {}

    sp._record_primary_secondary_step_equation(
        diagnostics,
        material=PronyConductivity.no_ip(0.02),
        sigma_background=0.01,
        dt=0.25,
    )

    metadata = diagnostics["primary_secondary_step_equation"]
    assert metadata["case_type"] == "noip"
    assert metadata["sigma"] == 0.02
    assert metadata["sigma_background"] == 0.01
    assert metadata["dt"] == 0.25
    assert metadata["lhs_operator"] == "K + R + M(sigma)/dt"


def test_record_primary_secondary_step_equation_writes_ip_metadata():
    sp = _load_pipeline_module()
    diagnostics = {}

    sp._record_primary_secondary_step_equation(
        diagnostics,
        material=PronyConductivity(
            sigma_inf=0.03,
            terms=[DebyeTerm(delta_sigma=0.006, tau=0.5)],
        ),
        sigma_background=0.01,
        dt=0.5,
    )

    metadata = diagnostics["primary_secondary_step_equation"]
    assert metadata["case_type"] == "ip"
    assert metadata["sigma0"] == pytest.approx(0.024)
    assert metadata["sigma_eff"] == pytest.approx(0.027)
    assert metadata["alpha"] == [0.5]
    assert metadata["beta"] == [0.5]
    assert metadata["adapter_backend"] == "dolfinx_primary_secondary"


def test_dolfinx_primary_secondary_bridge_passes_adapter_diagnostics(monkeypatch):
    sp = _load_pipeline_module()
    captured = {}
    diagnostics = {}

    def fake_interpolation(_msh, _spaces):
        return {"points": np.array([[0.0, 0.0, 0.0]])}

    def fake_adapters(_msh, _spaces, _materials, _operators, _config, _fem_points, **_kwargs):
        return {
            "secondary_state_initializer": object(),
            "secondary_state_loader": object(),
            "secondary_step_solver": object(),
            "secondary_receiver_projector": object(),
            "secondary_state_stepper": object(),
            "diagnostics": diagnostics,
        }

    class FakePrimarySecondaryForwardOperator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import atem3d.solvers as solvers

    monkeypatch.setattr(sp, "_nedelec_interpolation_points", fake_interpolation)
    monkeypatch.setattr(sp, "_make_dolfinx_primary_secondary_forward_adapters", fake_adapters)
    monkeypatch.setattr(solvers, "PrimarySecondaryForwardOperator", FakePrimarySecondaryForwardOperator)

    result = sp._make_dolfinx_primary_secondary_forward_operator(
        msh=object(),
        spaces={},
        materials={},
        operators={},
        config=SimpleNamespace(),
        primary=object(),
        receiver_locations=np.array([[0.0, 0.0, 0.0]]),
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.01),
        sigma_background=0.01,
        turnoff_time=1.0e-5,
        turnoff_steps=10,
    )

    assert captured["diagnostics"] is diagnostics
    assert result["diagnostics"] is diagnostics


def test_dolfinx_primary_secondary_bridge_does_not_use_ramp_solver_time_as_primary_floor(monkeypatch):
    sp = _load_pipeline_module()
    captured = {}

    def fake_interpolation(_msh, _spaces):
        return {"points": np.array([[0.0, 0.0, 0.0]])}

    def fake_adapters(_msh, _spaces, _materials, _operators, _config, _fem_points, **_kwargs):
        return {
            "secondary_state_initializer": object(),
            "secondary_state_loader": object(),
            "secondary_step_solver": object(),
            "secondary_receiver_projector": object(),
            "secondary_state_stepper": object(),
            "diagnostics": {},
        }

    class FakePrimarySecondaryForwardOperator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import atem3d.solvers as solvers

    monkeypatch.setattr(sp, "_nedelec_interpolation_points", fake_interpolation)
    monkeypatch.setattr(sp, "_make_dolfinx_primary_secondary_forward_adapters", fake_adapters)
    monkeypatch.setattr(solvers, "PrimarySecondaryForwardOperator", FakePrimarySecondaryForwardOperator)

    sp._make_dolfinx_primary_secondary_forward_operator(
        msh=object(),
        spaces={},
        materials={},
        operators={},
        config=SimpleNamespace(ramp_solver_t_min=1.0e-6),
        primary=object(),
        receiver_locations=np.array([[0.0, 0.0, 0.0]]),
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.01),
        sigma_background=0.01,
        turnoff_time=1.0e-5,
        turnoff_steps=10,
    )

    assert captured["primary_time_floor"] == 0.0


def test_primary_secondary_forward_branch_honors_stop_after_outputs(monkeypatch):
    sp = _load_pipeline_module()
    seen = {}

    class FakeComm:
        rank = 0

    class FakeMesh:
        comm = FakeComm()

    class FakeOperator:
        def forward(self, observation_times):
            seen["observation_times"] = np.asarray(observation_times, dtype=float)
            return np.asarray([[1.0, 2.0, 3.0]], dtype=float)

    def fake_generate_time_array(_config):
        return np.asarray([1.0e-5, 2.0e-5, 4.0e-5], dtype=float)

    monkeypatch.setattr(sp, "generate_time_array", fake_generate_time_array)
    monkeypatch.setattr(sp, "_primary_secondary_background_sigma", lambda _config: 0.01)
    monkeypatch.setattr(
        sp,
        "_primary_secondary_representative_material",
        lambda _config, debye=None: PronyConductivity.no_ip(0.01),
    )
    monkeypatch.setattr(sp, "_make_primary_secondary_background_sigma_function", lambda *_args: object())
    monkeypatch.setattr(sp, "_make_pipeline_empymod_primary_provider", lambda _config: object())
    monkeypatch.setattr(sp, "_primary_secondary_has_active_contrast", lambda _config: True)
    monkeypatch.setattr(sp, "assemble_operators", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        sp,
        "_make_dolfinx_primary_secondary_forward_operator",
        lambda *_args, **_kwargs: {"operator": FakeOperator(), "diagnostics": {}},
    )

    config = sp.PipelineConfig(
        source_term_mode="primary_secondary",
        stop_after_outputs=1,
        t_min=1.0e-5,
        t_max=4.0e-5,
    )
    result = sp._run_primary_secondary_fetd_forward(
        msh=FakeMesh(),
        facet_tags=object(),
        spaces={},
        materials={},
        config=config,
    )

    np.testing.assert_allclose(seen["observation_times"], [1.0e-5])
    np.testing.assert_allclose(result["times"], [1.0e-5])
    np.testing.assert_allclose(result["data"], [[1.0, 2.0, 3.0]])


def test_pipeline_primary_provider_receiver_dbdt_matches_reference():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 200.0, -0.1),
        source_end=(500.0, 200.0, -0.1),
        receiver=(0.0, -300.0, -0.1),
        source_current=10.0,
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        empymod_srcpts=11,
        ramp_off_time=0.0,
    )
    time = 1.0e-5

    primary = sp._make_pipeline_empymod_primary_provider(config)
    receiver_dbdt = primary.get_receiver_dBdt(time, [config.receiver])
    reference = sp.get_empymod_reference([time], config, mode="noip")
    dbzdt_index = reference["components"].index("dBzdt")

    assert receiver_dbdt[0, 2] == pytest.approx(reference["data"][0, dbzdt_index])


def test_primary_secondary_layered_ip_primary_keeps_noip_layer_resistivity():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        polarization="cole-cole",
        cole_rho0=33.33333333333334,
        cole_m=0.2,
        cole_tau=0.1,
        cole_c=0.6,
        cole_layer_top=100.0,
    )

    primary_config = sp._primary_secondary_empymod_primary_config(config)

    assert primary_config["resistivities"] == pytest.approx(
        [1.0e8, 100.0, 33.33333333333334]
    )


def test_masked_primary_provider_receiver_uses_primary_secondary_background_config(monkeypatch):
    sp = _load_pipeline_module()
    receiver = np.asarray([[0.0, -300.0, -0.1]], dtype=float)

    class FakeBaseProvider:
        def get_receiver_E(self, time_value, receivers):
            raise AssertionError("receiver E must use the validated pipeline reference path")

        def get_receiver_H(self, time_value, receivers):
            raise AssertionError("receiver H must use the validated pipeline reference path")

        def get_receiver_dBdt(self, time_value, receivers):
            raise AssertionError("receiver dBdt must use the validated pipeline reference path")

    captured = {}

    def primary_reference(times, config, mode="noip", **_kwargs):
        captured["times"] = np.asarray(times, dtype=float)
        captured["config"] = config
        captured["mode"] = mode
        return {
            "components": ["Ex", "Ey", "Hz", "dBzdt"],
            "data": np.asarray([[1.25, -0.5, 3.5e-10, -2.5e-8]], dtype=float),
        }

    config = sp.PipelineConfig(
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        polarization="cole-cole",
        cole_rho0=33.33333333333334,
        cole_m=0.2,
        cole_tau=0.1,
        cole_c=0.6,
        cole_layer_top=100.0,
    )
    monkeypatch.setattr(sp, "get_empymod_reference", primary_reference)
    wrapped = sp._MaskedPrimaryProvider(FakeBaseProvider(), config)

    np.testing.assert_allclose(
        wrapped.get_receiver_E(1.0e-5, receiver),
        [[1.25, -0.5, 0.0]],
    )
    np.testing.assert_allclose(
        wrapped.get_receiver_H(1.0e-5, receiver),
        [[0.0, 0.0, 3.5e-10]],
    )
    np.testing.assert_allclose(
        wrapped.get_receiver_dBdt(1.0e-5, receiver),
        [[0.0, 0.0, -2.5e-8]],
    )
    assert captured["mode"] == "noip"
    assert captured["config"].polarization == "none"
    assert captured["config"].receiver == pytest.approx(tuple(receiver[0]))
    assert captured["config"].layer_resistivities == pytest.approx(
        (100.0, 33.33333333333334)
    )


def test_masked_primary_provider_reuses_capped_sample_plan(monkeypatch):
    sp = _load_pipeline_module()

    class FakeBaseProvider:
        pass

    calls = {"indices": 0, "sampler": 0}
    original_indices = sp._primary_secondary_spatial_sample_indices

    def counting_indices(points, max_samples, config=None):
        calls["indices"] += 1
        return original_indices(points, max_samples, config)

    def sampler(points):
        calls["sampler"] += 1
        pts = np.asarray(points, dtype=float)
        scale = float(calls["sampler"])
        return scale * np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2]))

    selected = np.array(
        [
            [x, y, z]
            for x in (0.0, 1.0, 2.0)
            for y in (0.0, 1.0, 2.0)
            for z in (0.0, 1.0, 2.0)
        ],
        dtype=float,
    )
    config = sp.PipelineConfig(primary_secondary_max_primary_samples=8)
    wrapped = sp._MaskedPrimaryProvider(FakeBaseProvider(), config)
    monkeypatch.setattr(sp, "_primary_secondary_spatial_sample_indices", counting_indices)

    first = wrapped._sample_selected(sampler, selected)
    second = wrapped._sample_selected(sampler, selected.copy())

    assert calls["indices"] == 1
    assert calls["sampler"] == 2
    np.testing.assert_allclose(second, 2.0 * first)


def test_masked_primary_provider_samples_all_points_when_only_slightly_above_cap(monkeypatch):
    sp = _load_pipeline_module()

    class FakeBaseProvider:
        pass

    calls = {"indices": 0, "sample_sizes": []}

    def counting_indices(points, max_samples, config=None):
        calls["indices"] += 1
        return np.arange(int(max_samples), dtype=np.int64)

    def sampler(points):
        pts = np.asarray(points, dtype=float)
        calls["sample_sizes"].append(pts.shape[0])
        return np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2]))

    selected = np.array(
        [[float(i), 0.0, -100.0] for i in range(75)],
        dtype=float,
    )
    config = sp.PipelineConfig(primary_secondary_max_primary_samples=64)
    wrapped = sp._MaskedPrimaryProvider(FakeBaseProvider(), config)
    monkeypatch.setattr(sp, "_primary_secondary_spatial_sample_indices", counting_indices)

    values = wrapped._sample_selected(sampler, selected)

    assert calls["indices"] == 0
    assert calls["sample_sizes"] == [75]
    np.testing.assert_allclose(values, selected)


def test_primary_secondary_spatial_samples_prioritize_source_receiver_and_interface():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        source_start=(-500.0, 200.0, -0.1),
        source_end=(500.0, 200.0, -0.1),
        receiver=(0.0, -300.0, -0.1),
        layer_depths=(100.0,),
        layer_resistivities=(100.0, 33.33333333333334),
        polarization="cole-cole",
        cole_layer_top=100.0,
        cole_layer_bottom=float("inf"),
    )
    far_points = np.array(
        [
            [x, y, z]
            for x in np.linspace(-600.0, 600.0, 7)
            for y in np.linspace(-450.0, 450.0, 7)
            for z in np.linspace(-600.0, -150.0, 4)
        ],
        dtype=float,
    )
    anchor_points = np.array(
        [
            [0.0, -300.0, -105.0],
            [0.0, 200.0, -105.0],
            [500.0, 200.0, -105.0],
        ],
        dtype=float,
    )
    points = np.vstack([far_points, anchor_points])

    indices = sp._primary_secondary_spatial_sample_indices(points, 12, config)
    selected = points[indices]

    assert np.min(np.linalg.norm(selected[:, :2] - np.array([0.0, -300.0]), axis=1)) < 80.0
    assert np.min(np.abs(selected[:, 1] - 200.0)) < 80.0
    assert np.min(np.abs((-selected[:, 2]) - 100.0)) < 20.0


def test_primary_secondary_ip_state_stepper_returns_memory_samples(monkeypatch):
    sp = _load_pipeline_module()
    from atem3d.materials.prony import DebyeTerm, PronyConductivity

    class FakeArray:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float).reshape(-1)

        def scatter_forward(self):
            return None

    class FakePetscVec:
        def __init__(self, owner):
            self.owner = owner

        def norm(self):
            values = self.owner.x.array if hasattr(self.owner, "x") else self.owner.array
            return float(np.linalg.norm(values))

    class FakeX(FakeArray):
        @property
        def petsc_vec(self):
            return FakePetscVec(self)

    class FakeFunction:
        def __init__(self, values=None, name=None):
            if hasattr(values, "default_values"):
                values = values.default_values
            if values is None:
                values = np.zeros(3, dtype=float)
            self.x = FakeX(values)

        def __add__(self, other):
            return FakeFunction(self.x.array + other.x.array)

        def __radd__(self, other):
            return self.__add__(other)

        def __mul__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array * other.x.array)
            return FakeFunction(float(other) * self.x.array)

        def __rmul__(self, other):
            return self.__mul__(other)

        def __sub__(self, other):
            return FakeFunction(self.x.array - other.x.array)

        def interpolate(self, expression):
            if hasattr(expression, "x"):
                self.x.array[:] = expression.x.array
            else:
                self.x.array[:] = np.asarray(expression, dtype=float).reshape(-1)

    class FakeFEM:
        Function = FakeFunction

        @staticmethod
        def Expression(expression, *_args, **_kwargs):
            return expression

        class Constant:
            def __init__(self, _msh, value):
                self.value = float(value)

            def __mul__(self, other):
                return other * self.value

            def __rmul__(self, other):
                return other * self.value

    monkeypatch.setitem(sys.modules, "dolfinx.fem", FakeFEM)
    monkeypatch.setitem(sys.modules, "dolfinx", type("FakeDolfinx", (), {"fem": FakeFEM}))

    material = PronyConductivity(
        sigma_inf=0.03,
        terms=(DebyeTerm(delta_sigma=0.002, tau=0.1),),
    )
    config = sp.PipelineConfig(primary_secondary_current_correction="none")
    class FakeElement:
        @staticmethod
        def interpolation_points():
            return np.zeros((1, 3), dtype=float)

    class FakeSpace:
        element = FakeElement()
        default_values = np.zeros(3, dtype=float)

    spaces = {"V": FakeSpace(), "Q": object()}
    materials = {
        "sigma": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_initial": FakeFunction([0.028, 0.028, 0.028]),
        "sigma_infinity": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_background": FakeFunction([0.01, 0.01, 0.01]),
    }

    monkeypatch.setattr(
        sp,
        "_make_nedelec_rhs_interpolator_from_samples",
        lambda *_args, **_kwargs: lambda samples: FakeFunction(np.asarray(samples, dtype=float).reshape(-1)),
    )
    monkeypatch.setattr(
        sp,
        "_make_nedelec_solution_sampler_at_points",
        lambda *_args, **_kwargs: lambda function, template: np.asarray(function.x.array, dtype=float).reshape(np.asarray(template).shape),
    )
    monkeypatch.setattr(
        sp,
        "_make_dolfinx_secondary_step_solver",
        lambda *_args, **_kwargs: lambda rhs, sigma_eff, dt: np.asarray(rhs.x.array, dtype=float).reshape((1, 3)) * 0.0,
    )
    monkeypatch.setattr(
        sp,
        "_primary_secondary_corrected_trial_solution",
        lambda latest, trial, correct: trial,
    )
    monkeypatch.setattr(
        sp,
        "_solve_dc_secondary_field",
        lambda *_args, **_kwargs: {
            "Es0": FakeFunction([0.0, 0.0, 0.0]),
            "contrast_is_zero": False,
        },
    )
    monkeypatch.setattr(
        sp,
        "evaluate_receivers",
        lambda *_args, **_kwargs: {"Ex": 0.0, "Ey": 0.0, "dBzdt": 0.0},
    )
    monkeypatch.setattr(
        sp,
        "compute_dbdt",
        lambda *_args, **_kwargs: FakeFunction([0.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(sp, "_record_primary_secondary_step_equation", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        sp,
        "_make_secondary_receiver_projector_from_evaluate_receivers",
        lambda *args, **kwargs: object(),
    )

    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh=type("FakeMesh", (), {"comm": None})(),
        spaces=spaces,
        materials=materials,
        operators={},
        config=config,
        fem_points=np.asarray([[0.0, 0.0, -100.0]]),
        sigma_background=0.01,
        fem_cells=None,
        debye={"terms": material.terms, "delta_functions": ()},
    )
    init = adapters["secondary_state_initializer"](
        np.asarray([[1.0, 2.0, 3.0]]),
        material,
        0.01,
    )
    from atem3d.solvers.tdem_secondary import secondary_state_from_dc_initialization

    state = secondary_state_from_dc_initialization(init)

    stepped = adapters["secondary_state_stepper"](
        state,
        init.Ep0,
        np.asarray([[0.5, 1.0, 1.5]]),
        material,
        0.01,
        1.0e-3,
        1.0e-3,
    )

    assert len(stepped.chi) == len(material.terms)
    assert stepped.chi[0].shape == stepped.Es.shape


def test_primary_secondary_ip_state_stepper_uses_solved_secondary_samples(monkeypatch):
    sp = _load_pipeline_module()

    class FakeArray:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float).reshape(-1)

        def scatter_forward(self):
            return None

    class FakePetscVec:
        def __init__(self, owner):
            self.owner = owner

        def norm(self):
            values = self.owner.x.array if hasattr(self.owner, "x") else self.owner.array
            return float(np.linalg.norm(values))

        def set(self, value):
            self.owner.x.array[:] = float(value)

        def assemble(self):
            return None

    class FakeX(FakeArray):
        @property
        def petsc_vec(self):
            return FakePetscVec(self)

    class FakeFunction:
        def __init__(self, values=None, name=None):
            if hasattr(values, "default_values"):
                values = values.default_values
            if values is None:
                values = np.zeros(3, dtype=float)
            self.x = FakeX(values)

        def __add__(self, other):
            return FakeFunction(self.x.array + other.x.array)

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            return FakeFunction(self.x.array - other.x.array)

        def __rsub__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(other.x.array - self.x.array)
            return FakeFunction(np.asarray(other, dtype=float).reshape(-1) - self.x.array)

        def __mul__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array * other.x.array)
            return FakeFunction(float(other) * self.x.array)

        def __rmul__(self, other):
            return self.__mul__(other)

        def interpolate(self, expression):
            if hasattr(expression, "x"):
                self.x.array[:] = expression.x.array
            else:
                self.x.array[:] = np.asarray(expression, dtype=float).reshape(-1)

    class FakeFEM:
        Function = FakeFunction

        @staticmethod
        def Expression(expression, *_args, **_kwargs):
            return expression

        class Constant:
            def __init__(self, _msh, value):
                self.value = float(value)

            def __mul__(self, other):
                return other * self.value

            def __rmul__(self, other):
                return other * self.value

            def __sub__(self, other):
                return FakeFunction(np.full(3, self.value) - other.x.array)

            def __rsub__(self, other):
                return FakeFunction(other.x.array - np.full(3, self.value))

    class FakeElement:
        @staticmethod
        def interpolation_points():
            return np.zeros((1, 3), dtype=float)

    class FakeSpace:
        element = FakeElement()
        default_values = np.zeros(3, dtype=float)

    solved_samples = np.asarray([[4.0, -2.0, 1.0]], dtype=float)

    monkeypatch.setitem(sys.modules, "dolfinx.fem", FakeFEM)
    monkeypatch.setitem(sys.modules, "dolfinx", type("FakeDolfinx", (), {"fem": FakeFEM}))
    monkeypatch.setattr(
        sp,
        "_make_nedelec_rhs_interpolator_from_samples",
        lambda *_args, **_kwargs: lambda samples: FakeFunction(np.asarray(samples, dtype=float).reshape(-1)),
    )
    monkeypatch.setattr(
        sp,
        "_make_nedelec_solution_sampler_at_points",
        lambda *_args, **_kwargs: lambda function, template: np.asarray(function.x.array, dtype=float).reshape(-1, 3),
    )
    monkeypatch.setattr(
        sp,
        "_make_dolfinx_secondary_step_solver",
        lambda *_args, **_kwargs: lambda rhs, sigma_eff, dt: solved_samples.copy(),
    )
    monkeypatch.setattr(
        sp,
        "_primary_secondary_corrected_trial_solution",
        lambda previous, trial, correct: trial,
    )
    monkeypatch.setattr(
        sp,
        "_solve_dc_secondary_field",
        lambda *_args, **_kwargs: {
            "Es0": FakeFunction([0.0, 0.0, 0.0]),
            "contrast_is_zero": False,
        },
    )
    monkeypatch.setattr(sp, "evaluate_receivers", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sp, "compute_dbdt", lambda *_args, **_kwargs: FakeFunction([0.0, 0.0, 0.0]))
    monkeypatch.setattr(sp, "_record_primary_secondary_step_equation", lambda *args, **kwargs: {})
    monkeypatch.setattr(sp, "_make_secondary_receiver_projector_from_evaluate_receivers", lambda *args, **kwargs: object())

    material = PronyConductivity(
        sigma_inf=0.03,
        terms=(DebyeTerm(delta_sigma=0.002, tau=0.1),),
    )
    spaces = {"V": FakeSpace(), "Q": object()}
    materials = {
        "sigma": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_initial": FakeFunction([0.028, 0.028, 0.028]),
        "sigma_infinity": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_background": FakeFunction([0.01, 0.01, 0.01]),
    }
    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh=type("FakeMesh", (), {"comm": None})(),
        spaces=spaces,
        materials=materials,
        operators={},
        config=sp.PipelineConfig(primary_secondary_current_correction="none"),
        fem_points=np.asarray([[0.0, 0.0, -100.0]]),
        sigma_background=0.01,
        fem_cells=None,
        debye={"terms": material.terms, "delta_functions": ()},
    )
    init = adapters["secondary_state_initializer"](np.asarray([[1.0, 2.0, 3.0]]), material, 0.01)
    from atem3d.solvers.tdem_secondary import secondary_state_from_dc_initialization

    stepped = adapters["secondary_state_stepper"](
        secondary_state_from_dc_initialization(init),
        init.Ep0,
        np.asarray([[0.5, 1.0, 1.5]]),
        material,
        0.01,
        1.0e-3,
        1.0e-3,
    )

    np.testing.assert_allclose(stepped.Es, solved_samples)


def test_primary_secondary_adapter_records_transient_background_mode(monkeypatch):
    sp = _load_pipeline_module()
    from atem3d.materials.prony import DebyeTerm, PronyConductivity

    class FakeArray:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float).reshape(-1)

        def scatter_forward(self):
            return None

    class FakePetscVec:
        def __init__(self, owner):
            self.owner = owner

        def norm(self):
            values = self.owner.x.array if hasattr(self.owner, "x") else self.owner.array
            return float(np.linalg.norm(values))

    class FakeX(FakeArray):
        @property
        def petsc_vec(self):
            return FakePetscVec(self)

    class FakeFunction:
        def __init__(self, values=None, name=None):
            if hasattr(values, "default_values"):
                values = values.default_values
            if values is None:
                values = np.zeros(3, dtype=float)
            self.x = FakeX(values)

        def __add__(self, other):
            return FakeFunction(self.x.array + other.x.array)

        def __sub__(self, other):
            return FakeFunction(self.x.array - other.x.array)

        def __mul__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array * other.x.array)
            return FakeFunction(float(other) * self.x.array)

        def __rmul__(self, other):
            return self.__mul__(other)

        def interpolate(self, expression):
            self.x.array[:] = expression.x.array if hasattr(expression, "x") else np.asarray(expression, dtype=float)

    class FakeFEM:
        Function = FakeFunction

        @staticmethod
        def Expression(expression, *_args, **_kwargs):
            return expression

        class Constant:
            def __init__(self, _msh, value):
                self.value = float(value)

            def __mul__(self, other):
                return other * self.value

            def __rmul__(self, other):
                return other * self.value

            def __sub__(self, other):
                return FakeFunction(np.full(3, self.value) - other.x.array)

            def __rsub__(self, other):
                return FakeFunction(other.x.array - np.full(3, self.value))

    monkeypatch.setitem(sys.modules, "dolfinx.fem", FakeFEM)
    monkeypatch.setitem(sys.modules, "dolfinx", type("FakeDolfinx", (), {"fem": FakeFEM}))
    monkeypatch.setattr(
        sp,
        "_make_nedelec_rhs_interpolator_from_samples",
        lambda *_args, **_kwargs: lambda samples: FakeFunction(np.asarray(samples, dtype=float).reshape(-1)),
    )
    monkeypatch.setattr(
        sp,
        "_make_nedelec_solution_sampler_at_points",
        lambda *_args, **_kwargs: lambda function, template: np.asarray(function.x.array, dtype=float).reshape(np.asarray(template).shape),
    )
    monkeypatch.setattr(sp, "_make_dolfinx_secondary_step_solver", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sp, "_make_secondary_receiver_projector_from_evaluate_receivers", lambda *args, **kwargs: object())

    material = PronyConductivity(
        sigma_inf=0.03,
        terms=(DebyeTerm(delta_sigma=0.002, tau=0.1),),
    )

    class FakeElement:
        @staticmethod
        def interpolation_points():
            return np.zeros((1, 3), dtype=float)

    class FakeSpace:
        element = FakeElement()
        default_values = np.zeros(3, dtype=float)

    materials = {
        "sigma": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_initial": FakeFunction([0.028, 0.028, 0.028]),
        "sigma_infinity": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_background": FakeFunction([0.01, 0.02, 0.03]),
    }

    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh=type("FakeMesh", (), {"comm": None})(),
        spaces={"V": FakeSpace(), "Q": object()},
        materials=materials,
        operators={},
        config=sp.PipelineConfig(primary_secondary_transient_background_mode="scalar_top"),
        fem_points=np.asarray([[0.0, 0.0, -100.0]]),
        sigma_background=0.01,
        fem_cells=None,
        debye={"terms": material.terms, "delta_functions": ()},
    )

    assert adapters["diagnostics"]["transient_background_mode"] == "scalar_top"


def test_primary_secondary_scalar_top_mode_uses_scalar_background_for_dc_initialization(monkeypatch):
    sp = _load_pipeline_module()
    captured = {}

    class FakeArray:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float).reshape(-1)

        def scatter_forward(self):
            return None

    class FakePetscVec:
        def __init__(self, owner):
            self.owner = owner

        def norm(self):
            values = self.owner.x.array if hasattr(self.owner, "x") else self.owner.array
            return float(np.linalg.norm(values))

    class FakeX(FakeArray):
        @property
        def petsc_vec(self):
            return FakePetscVec(self)

    class FakeFunction:
        def __init__(self, values=None, name=None):
            if hasattr(values, "default_values"):
                values = values.default_values
            if values is None:
                values = np.zeros(3, dtype=float)
            self.x = FakeX(values)

        def __add__(self, other):
            return FakeFunction(self.x.array + other.x.array)

        def __sub__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array - other.x.array)
            return FakeFunction(self.x.array - float(other))

        def __mul__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array * other.x.array)
            return FakeFunction(float(other) * self.x.array)

        def __rmul__(self, other):
            return self.__mul__(other)

        def interpolate(self, expression):
            self.x.array[:] = expression.x.array if hasattr(expression, "x") else np.asarray(expression, dtype=float)

    class FakeConstant:
        def __init__(self, _msh, value):
            self.value = float(value)

        def __mul__(self, other):
            return other * self.value

        def __rmul__(self, other):
            return other * self.value

        def __sub__(self, other):
            return FakeFunction(np.full(3, self.value) - other.x.array)

        def __rsub__(self, other):
            return FakeFunction(other.x.array - np.full(3, self.value))

    class FakeFEM:
        Function = FakeFunction
        Constant = FakeConstant

        @staticmethod
        def Expression(expression, *_args, **_kwargs):
            return expression

    class FakeElement:
        @staticmethod
        def interpolation_points():
            return np.zeros((1, 3), dtype=float)

    class FakeSpace:
        element = FakeElement()
        default_values = np.zeros(3, dtype=float)

    monkeypatch.setitem(sys.modules, "dolfinx.fem", FakeFEM)
    monkeypatch.setitem(sys.modules, "dolfinx", type("FakeDolfinx", (), {"fem": FakeFEM}))
    monkeypatch.setattr(
        sp,
        "_make_nedelec_rhs_interpolator_from_samples",
        lambda *_args, **_kwargs: lambda samples: FakeFunction(np.asarray(samples, dtype=float).reshape(-1)),
    )
    monkeypatch.setattr(
        sp,
        "_make_nedelec_solution_sampler_at_points",
        lambda *_args, **_kwargs: lambda function, template: np.asarray(function.x.array, dtype=float).reshape(np.asarray(template).shape),
    )
    monkeypatch.setattr(sp, "_make_dolfinx_secondary_step_solver", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sp, "_make_secondary_receiver_projector_from_evaluate_receivers", lambda *args, **kwargs: object())
    monkeypatch.setattr(sp, "evaluate_receivers", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sp, "compute_dbdt", lambda *_args, **_kwargs: FakeFunction([0.0, 0.0, 0.0]))

    def fake_solve_dc(*_args, **kwargs):
        captured["sigma_background"] = kwargs["sigma_background"]
        captured["sigma_lhs"] = kwargs.get("sigma_lhs")
        return {
            "Es0": FakeFunction([0.0, 0.0, 0.0]),
            "contrast_is_zero": False,
        }

    monkeypatch.setattr(sp, "_solve_dc_secondary_field", fake_solve_dc)

    from atem3d.materials.prony import DebyeTerm, PronyConductivity

    material = PronyConductivity(
        sigma_inf=0.03,
        terms=(DebyeTerm(delta_sigma=0.002, tau=0.1),),
    )
    materials = {
        "sigma": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_initial": FakeFunction([0.028, 0.028, 0.028]),
        "sigma_infinity": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_background": FakeFunction([0.01, 0.02, 0.03]),
    }
    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh=type("FakeMesh", (), {"comm": None})(),
        spaces={"V": FakeSpace(), "Q": object()},
        materials=materials,
        operators={},
        config=sp.PipelineConfig(primary_secondary_transient_background_mode="scalar_top"),
        fem_points=np.asarray([[0.0, 0.0, -100.0]]),
        sigma_background=0.01,
        fem_cells=None,
        debye={"terms": material.terms, "delta_functions": ()},
    )

    adapters["secondary_state_initializer"](np.asarray([[1.0, 2.0, 3.0]]), material, 0.01)

    assert isinstance(captured["sigma_background"], FakeConstant)
    assert captured["sigma_background"].value == pytest.approx(0.01)


def test_primary_secondary_ip_increment_mode_uses_initial_conductivity_as_background(monkeypatch):
    sp = _load_pipeline_module()
    captured = {}

    class FakeArray:
        def __init__(self, values):
            self.array = np.asarray(values, dtype=float).reshape(-1)

        def scatter_forward(self):
            return None

    class FakePetscVec:
        def __init__(self, owner):
            self.owner = owner

        def norm(self):
            values = self.owner.x.array if hasattr(self.owner, "x") else self.owner.array
            return float(np.linalg.norm(values))

    class FakeX(FakeArray):
        @property
        def petsc_vec(self):
            return FakePetscVec(self)

    class FakeFunction:
        def __init__(self, values=None, name=None):
            if hasattr(values, "default_values"):
                values = values.default_values
            if values is None:
                values = np.zeros(3, dtype=float)
            self.x = FakeX(values)

        def __add__(self, other):
            return FakeFunction(self.x.array + other.x.array)

        def __sub__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array - other.x.array)
            return FakeFunction(self.x.array - float(other))

        def __mul__(self, other):
            if hasattr(other, "x"):
                return FakeFunction(self.x.array * other.x.array)
            return FakeFunction(float(other) * self.x.array)

        def __rmul__(self, other):
            return self.__mul__(other)

        def interpolate(self, expression):
            self.x.array[:] = expression.x.array if hasattr(expression, "x") else np.asarray(expression, dtype=float)

    class FakeFEM:
        Function = FakeFunction

        @staticmethod
        def Expression(expression, *_args, **_kwargs):
            return expression

    class FakeElement:
        @staticmethod
        def interpolation_points():
            return np.zeros((1, 3), dtype=float)

    class FakeSpace:
        element = FakeElement()
        default_values = np.zeros(3, dtype=float)

    monkeypatch.setitem(sys.modules, "dolfinx.fem", FakeFEM)
    monkeypatch.setitem(sys.modules, "dolfinx", type("FakeDolfinx", (), {"fem": FakeFEM}))
    monkeypatch.setattr(
        sp,
        "_make_nedelec_rhs_interpolator_from_samples",
        lambda *_args, **_kwargs: lambda samples: FakeFunction(np.asarray(samples, dtype=float).reshape(-1)),
    )
    monkeypatch.setattr(
        sp,
        "_make_nedelec_solution_sampler_at_points",
        lambda *_args, **_kwargs: lambda function, template: np.asarray(function.x.array, dtype=float).reshape(np.asarray(template).shape),
    )
    monkeypatch.setattr(sp, "_make_dolfinx_secondary_step_solver", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sp, "_make_secondary_receiver_projector_from_evaluate_receivers", lambda *args, **kwargs: object())
    monkeypatch.setattr(sp, "evaluate_receivers", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(sp, "compute_dbdt", lambda *_args, **_kwargs: FakeFunction([0.0, 0.0, 0.0]))

    def fake_solve_dc(*_args, **kwargs):
        captured["sigma_background"] = kwargs["sigma_background"]
        captured["sigma_lhs"] = kwargs.get("sigma_lhs")
        return {
            "Es0": FakeFunction([0.0, 0.0, 0.0]),
            "contrast_is_zero": False,
        }

    monkeypatch.setattr(sp, "_solve_dc_secondary_field", fake_solve_dc)

    material = PronyConductivity(
        sigma_inf=0.03,
        terms=(DebyeTerm(delta_sigma=0.002, tau=0.1),),
    )
    materials = {
        "sigma": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_initial": FakeFunction([0.0225, 0.0225, 0.0225]),
        "sigma_infinity": FakeFunction([0.03, 0.03, 0.03]),
        "sigma_background": FakeFunction([0.01, 0.01, 0.01]),
    }
    adapters = sp._make_dolfinx_primary_secondary_forward_adapters(
        msh=type("FakeMesh", (), {"comm": None})(),
        spaces={"V": FakeSpace(), "Q": object()},
        materials=materials,
        operators={},
        config=sp.PipelineConfig(
            primary_secondary_dc_conductivity_mode="sigma_infinity",
            primary_secondary_transient_background_mode="ip_increment",
        ),
        fem_points=np.asarray([[0.0, 0.0, -100.0]]),
        sigma_background=0.01,
        fem_cells=None,
        debye={"terms": material.terms, "delta_functions": ()},
    )

    adapters["secondary_state_initializer"](np.asarray([[1.0, 2.0, 3.0]]), material, 0.01)

    assert captured["sigma_background"] is materials["sigma_initial"]
    assert captured["sigma_lhs"] is materials["sigma_initial"]
    assert adapters["diagnostics"]["transient_background_mode"] == "ip_increment"
    assert adapters["diagnostics"]["nominal_contrast"] == pytest.approx(0.0075)
