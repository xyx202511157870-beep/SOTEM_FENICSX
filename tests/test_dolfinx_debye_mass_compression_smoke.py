from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("dolfinx.fem")
pytest.importorskip("dolfinx.mesh")


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_debye_mass_compression_smoke",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _destroy_operators(operators):
    matrices = [operators.get(name) for name in ("K", "M", "M_inf", "B_robin", "M_debye_shared_basis")]
    matrices.extend(operators.get("M_debye", ()))
    for matrix in matrices:
        if matrix is not None:
            matrix.destroy()


def test_p2_shared_basis_matches_per_term_operators_on_small_mesh():
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    tdim = msh.topology.dim
    n_local_cells = msh.topology.index_map(tdim).size_local
    cell_tags = mesh.meshtags(
        msh,
        tdim,
        np.arange(n_local_cells, dtype=np.int32),
        np.full(n_local_cells, sp.PHYS_EARTH, dtype=np.int32),
    )
    config = sp.PipelineConfig(
        nedelec_order=2,
        outer_boundary_mode="natural",
        cole_layer_top=-2.0,
        cole_layer_bottom=2.0,
    )
    spaces = sp.build_function_spaces(msh, config)
    terms = [
        sp.DebyeTerm(delta_sigma=2.0e-3, tau=1.0e-3),
        sp.DebyeTerm(delta_sigma=3.5e-3, tau=2.0e-2),
        sp.DebyeTerm(delta_sigma=1.0e-3, tau=4.0e-1),
    ]
    fit = sp.DebyeFit(
        sigma_infinity=1.2e-2,
        terms=terms,
        frequencies=np.empty(0),
        target_sigma=np.empty(0),
        fitted_sigma=np.empty(0),
        relative_l2=0.0,
    )
    debye = sp._build_debye_materials(msh, cell_tags, spaces, fit, config)
    Q = spaces["Q"]
    sigma = fem.Function(Q)
    sigma.x.array[:] = 8.0e-3
    sigma.x.scatter_forward()
    sigma_inf = fem.Function(Q)
    sigma_inf.x.array[:] = fit.sigma_infinity
    sigma_inf.x.scatter_forward()
    mu_inv = fem.Function(Q)
    mu_inv.x.array[:] = 1.0
    mu_inv.x.scatter_forward()
    materials = {"sigma": sigma, "sigma_infinity": sigma_inf, "mu_inv": mu_inv}

    fallback_debye = {key: value for key, value in debye.items() if key != "shared_mass_basis"}
    shared_operators = sp.assemble_operators(msh, spaces, materials, None, config, debye=debye)
    fallback_operators = sp.assemble_operators(msh, spaces, materials, None, config, debye=fallback_debye)
    shared_effective = fallback_effective = shared_rhs = fallback_rhs = difference = None
    try:
        assert shared_operators["M_debye"] == []
        assert shared_operators["M_debye_shared_basis"] is not None
        assert shared_operators["M_debye_shared_weights"] == tuple(term.delta_sigma for term in terms)
        assert len(fallback_operators["M_debye"]) == len(terms)
        assert fallback_operators["M_debye_shared_basis"] is None

        dt = 2.5e-3
        shared_effective = sp._matrix_for_effective_conductivity(shared_operators, debye, dt)
        fallback_effective = sp._matrix_for_effective_conductivity(fallback_operators, fallback_debye, dt)
        difference = shared_effective.copy()
        difference.axpy(-1.0, fallback_effective)
        difference.assemble()
        assert difference.norm() <= 1.0e-12 * max(1.0, fallback_effective.norm())
        difference.destroy()
        difference = None

        E_old = fem.Function(spaces["V"])
        E_old.x.array[:] = np.linspace(-0.75, 0.5, E_old.x.array.size)
        E_old.x.scatter_forward()
        memories = []
        for index, _term in enumerate(terms):
            memory = fem.Function(spaces["V"])
            memory.x.array[:] = np.linspace(0.1 + index, 0.4 + index, memory.x.array.size)
            memory.x.scatter_forward()
            memories.append(memory)
        shared_rhs = sp._assemble_history_rhs(shared_operators, debye, memories, E_old, dt)
        fallback_rhs = sp._assemble_history_rhs(fallback_operators, fallback_debye, memories, E_old, dt)
        shared_rhs.axpy(-1.0, fallback_rhs)
        assert shared_rhs.norm() <= 1.0e-12 * max(1.0, fallback_rhs.norm())
    finally:
        for item in (difference, shared_effective, fallback_effective, shared_rhs, fallback_rhs):
            if item is not None:
                item.destroy()
        _destroy_operators(shared_operators)
        _destroy_operators(fallback_operators)
