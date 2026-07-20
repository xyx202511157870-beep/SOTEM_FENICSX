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
        "sotem_pipeline_for_debye_mass_compression_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Vec:
    created: list["_Vec"] = []

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float).copy()
        self.destroy_count = 0
        type(self).created.append(self)

    def duplicate(self):
        return _Vec(np.zeros_like(self.values))

    def axpy(self, coefficient, other):
        self.values += float(coefficient) * other.values

    def scale(self, coefficient):
        self.values *= float(coefficient)

    def set(self, value):
        self.values[:] = float(value)

    def destroy(self):
        self.destroy_count += 1


class _Matrix:
    def __init__(self, values, *, fail_mult=False):
        self.values = np.asarray(values, dtype=float).copy()
        self.fail_mult = bool(fail_mult)
        self.destroy_count = 0
        self.assemble_count = 0

    def copy(self):
        return _Matrix(self.values)

    def axpy(self, coefficient, other, **_kwargs):
        self.values += float(coefficient) * other.values

    def assemble(self):
        self.assemble_count += 1

    def mult(self, source, target):
        if self.fail_mult:
            raise RuntimeError("matrix action failed")
        target.values[:] = self.values @ source.values

    def destroy(self):
        self.destroy_count += 1


def _field(values):
    return SimpleNamespace(x=SimpleNamespace(petsc_vec=_Vec(values)))


def _install_fake_petsc(monkeypatch):
    petsc4py = ModuleType("petsc4py")
    petsc4py.PETSc = SimpleNamespace(
        Mat=SimpleNamespace(Structure=SimpleNamespace(SAME_NONZERO_PATTERN=object()))
    )
    monkeypatch.setitem(sys.modules, "petsc4py", petsc4py)


def _terms(sp):
    return (
        sp.DebyeTerm(delta_sigma=0.2, tau=0.5),
        sp.DebyeTerm(delta_sigma=0.35, tau=2.0),
    )


def test_shared_basis_effective_matrix_is_exact_weighted_sum(monkeypatch):
    sp = _load_pipeline_module()
    _install_fake_petsc(monkeypatch)
    terms = _terms(sp)
    mass_inf = _Matrix([[5.0, 0.5], [0.5, 4.0]])
    basis = _Matrix([[2.0, 0.25], [0.25, 3.0]])
    debye = {"terms": terms}
    operators = {
        "M_inf": mass_inf,
        "M_debye": [],
        "M_debye_shared_basis": basis,
        "M_debye_shared_weights": (0.2, 0.35),
    }

    result = sp._matrix_for_effective_conductivity(operators, debye, 0.25)

    betas = [sp._debye_backward_euler_coefficients(term, 0.25)[1] for term in terms]
    expected = mass_inf.values - sum(beta * weight for beta, weight in zip(betas, (0.2, 0.35))) * basis.values
    np.testing.assert_allclose(result.values, expected)
    assert result.assemble_count == 1


def test_absent_shared_metadata_preserves_arbitrary_per_term_fallback(monkeypatch):
    sp = _load_pipeline_module()
    _install_fake_petsc(monkeypatch)
    terms = _terms(sp)
    mass_inf = _Matrix([[5.0, 0.5], [0.5, 4.0]])
    first = _Matrix([[0.4, 0.1], [0.1, 0.0]])
    second = _Matrix([[0.0, 0.0], [0.0, 1.2]])
    operators = {"M_inf": mass_inf, "M_debye": [first, second]}

    result = sp._matrix_for_effective_conductivity(operators, {"terms": terms}, 0.25)

    betas = [sp._debye_backward_euler_coefficients(term, 0.25)[1] for term in terms]
    expected = mass_inf.values - betas[0] * first.values - betas[1] * second.values
    np.testing.assert_allclose(result.values, expected)


def test_shared_basis_history_rhs_matches_per_term_matrices(monkeypatch):
    sp = _load_pipeline_module()
    _install_fake_petsc(monkeypatch)
    terms = _terms(sp)
    mass_inf = _Matrix([[5.0, 0.5], [0.5, 4.0]])
    basis = _Matrix([[2.0, 0.25], [0.25, 3.0]])
    weights = (0.2, 0.35)
    E_old = _field([1.5, -0.5])
    memories = [_field([0.25, 1.0]), _field([-0.75, 0.5])]
    debye = {"terms": terms}

    shared = sp._assemble_history_rhs(
        {
            "M_inf": mass_inf,
            "M_debye": [],
            "M_debye_shared_basis": basis,
            "M_debye_shared_weights": weights,
        },
        debye,
        memories,
        E_old,
        0.25,
    )
    fallback = sp._assemble_history_rhs(
        {
            "M_inf": mass_inf,
            "M_debye": [_Matrix(weight * basis.values) for weight in weights],
        },
        debye,
        memories,
        E_old,
        0.25,
    )

    np.testing.assert_allclose(shared.values, fallback.values)


def test_noip_returns_existing_mass_without_inspecting_compression_metadata():
    sp = _load_pipeline_module()
    mass = object()

    assert sp._matrix_for_effective_conductivity(
        {"M": mass, "M_debye_shared_weights": (1.0,)},
        None,
        0.25,
    ) is mass


def test_history_rhs_destroys_all_owned_vectors_when_shared_action_fails(monkeypatch):
    sp = _load_pipeline_module()
    _install_fake_petsc(monkeypatch)
    _Vec.created = []
    E_old = _field([1.0, 2.0])
    memories = [_field([3.0, 4.0])]
    preexisting = set(_Vec.created)

    with pytest.raises(RuntimeError, match="matrix action failed"):
        sp._assemble_history_rhs(
            {
                "M_inf": _Matrix(np.eye(2)),
                "M_debye": [],
                "M_debye_shared_basis": _Matrix(np.eye(2), fail_mult=True),
                "M_debye_shared_weights": (0.2,),
            },
            {"terms": (sp.DebyeTerm(delta_sigma=0.2, tau=1.0),)},
            memories,
            E_old,
            0.25,
        )

    owned = [vec for vec in _Vec.created if vec not in preexisting]
    assert len(owned) == 3
    assert all(vec.destroy_count == 1 for vec in owned)


def test_canonical_material_builder_marks_only_explicit_shared_basis(monkeypatch):
    sp = _load_pipeline_module()

    class _Array:
        def __init__(self, size):
            self.array = np.zeros(size, dtype=float)

        def scatter_forward(self):
            pass

    class _Function:
        def __init__(self, space, name):
            self.name = name
            self.x = _Array(space.size)

    fem = ModuleType("dolfinx.fem")
    fem.Function = _Function
    dolfinx = ModuleType("dolfinx")
    dolfinx.fem = fem
    monkeypatch.setitem(sys.modules, "dolfinx", dolfinx)
    monkeypatch.setitem(sys.modules, "dolfinx.fem", fem)
    monkeypatch.setattr(sp, "_polarizable_earth_cells", lambda *_args: np.array([1, 3], dtype=np.int32))
    monkeypatch.setattr(
        sp,
        "_assign_dg0_by_cell",
        lambda function, cells, value: function.x.array.__setitem__(cells, value),
    )
    terms = _terms(sp)
    fit = SimpleNamespace(terms=terms)

    debye = sp._build_debye_materials(
        object(),
        object(),
        {"Q": SimpleNamespace(size=5)},
        fit,
        SimpleNamespace(cole_layer_top=0.0, cole_layer_bottom=100.0),
    )

    shared = debye["shared_mass_basis"]
    assert shared["kind"] == "polarizable_cell_indicator"
    assert shared["exact"] is True
    assert shared["weights"] == tuple(term.delta_sigma for term in terms)
    np.testing.assert_array_equal(shared["function"].x.array, [0.0, 1.0, 0.0, 1.0, 0.0])


def test_malformed_explicit_shared_basis_is_rejected_instead_of_approximated(monkeypatch):
    sp = _load_pipeline_module()
    _install_fake_petsc(monkeypatch)
    terms = _terms(sp)
    copied = _Matrix(np.eye(2))
    mass_inf = SimpleNamespace(copy=lambda: copied)

    with pytest.raises(ValueError, match="shared Debye mass weights must match terms"):
        sp._matrix_for_effective_conductivity(
            {
                "M_inf": mass_inf,
                "M_debye": [],
                "M_debye_shared_basis": _Matrix(np.eye(2)),
                "M_debye_shared_weights": (0.2,),
            },
            {"terms": terms},
            0.25,
        )

    assert copied.destroy_count == 0
