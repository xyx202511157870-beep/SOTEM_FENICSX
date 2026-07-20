from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_source_gradient_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _relative_endpoint_residual(sp, msh, scalar_space, edge_space, source_vec, config):
    from dolfinx.fem import petsc as fem_petsc

    gradient = fem_petsc.discrete_gradient(scalar_space, edge_space)
    gradient.assemble()
    endpoint = sp._build_endpoint_scalar_load(msh, scalar_space, config, use_unit_current=True)
    residual = endpoint.duplicate()
    residual.set(0.0)
    gradient.multTranspose(source_vec, residual)
    residual.axpy(-1.0, endpoint)
    relative = float(residual.norm() / endpoint.norm())
    residual.destroy()
    endpoint.destroy()
    gradient.destroy()
    return relative


def _relative_vector_difference(left, right):
    difference = left.copy()
    difference.axpy(-1.0, right)
    relative = float(difference.norm() / max(left.norm(), right.norm()))
    difference.destroy()
    return relative


def _interior_segment(sp, msh, cell, variant):
    coords = sp._cell_geometry(msh, cell)
    barycentric_pairs = (
        ([0.55, 0.15, 0.20, 0.10], [0.15, 0.50, 0.10, 0.25]),
        ([0.10, 0.55, 0.15, 0.20], [0.25, 0.10, 0.50, 0.15]),
        ([0.20, 0.10, 0.55, 0.15], [0.15, 0.25, 0.10, 0.50]),
    )
    start_weights, end_weights = barycentric_pairs[int(variant) % len(barycentric_pairs)]
    return (
        tuple(np.asarray(start_weights) @ coords),
        tuple(np.asarray(end_weights) @ coords),
    )


def test_order_two_manual_line_transform_satisfies_de_rham_identity_for_all_real_cell_permutations():
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    spaces = sp.build_function_spaces(msh, sp.PipelineConfig(nedelec_order=2))
    msh.topology.create_entity_permutations()
    permutations = np.asarray(msh.topology.get_cell_permutation_info())
    residuals = []
    reversal_errors = []
    for cell, permutation in enumerate(permutations):
        start, end = _interior_segment(sp, msh, cell, int(permutation))
        config = sp.PipelineConfig(
            nedelec_order=2,
            source_start=start,
            source_end=end,
            source_mode="manual_line",
            source_quadrature_points=3,
            source_projection_mode="raw",
        )
        forward = sp._build_manual_line_source(msh, spaces, config)
        residuals.append(
            _relative_endpoint_residual(
                sp,
                msh,
                spaces["S"],
                spaces["V"],
                forward["vector"],
                config,
            )
        )
        reverse_config = sp.PipelineConfig(
            nedelec_order=2,
            source_start=end,
            source_end=start,
            source_mode="manual_line",
            source_quadrature_points=3,
            source_projection_mode="raw",
        )
        reverse = sp._build_manual_line_source(msh, spaces, reverse_config)
        reverse["vector"].scale(-1.0)
        reversal_errors.append(_relative_vector_difference(forward["vector"], reverse["vector"]))
        forward["vector"].destroy()
        reverse["vector"].destroy()

    assert len(set(int(value) for value in permutations)) > 1
    assert max(residuals) < 1.0e-11
    assert max(reversal_errors) < 1.0e-12


def test_order_two_manual_line_projection_satisfies_complete_p2_gradient_space():
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    msh.topology.create_entity_permutations()
    permutations = np.asarray(msh.topology.get_cell_permutation_info())
    cell = int(np.argmax(permutations))
    source_start, source_end = _interior_segment(sp, msh, cell, int(permutations[cell]))
    config = sp.PipelineConfig(
        nedelec_order=2,
        source_start=source_start,
        source_end=source_end,
        source_mode="manual_line",
        source_quadrature_points=3,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_it=1000,
    )
    spaces = sp.build_function_spaces(msh, config)
    raw_source = sp._build_manual_line_source(msh, spaces, config)
    raw_relative = _relative_endpoint_residual(
        sp,
        msh,
        spaces["S"],
        spaces["V"],
        raw_source["vector"],
        config,
    )
    source = sp.build_source(msh, spaces, config)
    gate = sp._require_source_projection_gate(source)
    complete_p2 = fem.functionspace(msh, ("Lagrange", 2))

    relative = _relative_endpoint_residual(
        sp,
        msh,
        complete_p2,
        spaces["V"],
        source["vector"],
        config,
    )

    assert raw_relative < 1.0e-11
    assert relative < 1.0e-11
    assert _relative_vector_difference(source["vector"], raw_source["vector"]) < 1.0e-11
    assert spaces["S"].element.basix_element.degree == 2
    assert source["projection_diagnostics"]["after_residual"] / source["projection_diagnostics"]["endpoint_norm"] < 1.0e-11
    assert source["projection_diagnostics"]["correction_l2_over_raw"] < 1.0e-11
    assert gate["passed"] is True
    assert source["projection_diagnostics"]["gradient_shape"] == [
        spaces["V"].dofmap.index_map.size_global,
        spaces["S"].dofmap.index_map.size_global,
    ]
    assert source["projection_diagnostics"]["projection_matrix_shape"] == [
        spaces["S"].dofmap.index_map.size_global,
        spaces["S"].dofmap.index_map.size_global,
    ]
    raw_source["vector"].destroy()
    source["vector"].destroy()


def test_order_one_single_cell_source_passes_fixed_projection_gate():
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    source_start, source_end = _interior_segment(sp, msh, 0, 0)
    config = sp.PipelineConfig(
        nedelec_order=1,
        source_start=source_start,
        source_end=source_end,
        source_mode="manual_line",
        source_quadrature_points=2,
        rtol=1.0e-11,
        atol=1.0e-13,
        max_it=1000,
    )
    spaces = sp.build_function_spaces(msh, config)
    source = sp.build_source(msh, spaces, config)

    gate = sp._require_source_projection_gate(source)

    assert spaces["S"].element.basix_element.degree == 1
    assert gate["passed"] is True
    source["vector"].destroy()


@pytest.mark.parametrize("nedelec_order", [1, 2])
def test_cross_cell_exact_segments_satisfy_raw_de_rham_and_reverse_orientation(nedelec_order):
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)
    config = sp.PipelineConfig(
        nedelec_order=nedelec_order,
        source_start=(0.2, 0.285, 0.389),
        source_end=(0.8, 0.285, 0.389),
        source_mode="manual_line",
        source_quadrature_points=33,
        source_projection_mode="raw",
        rtol=1.0e-11,
        atol=1.0e-13,
        max_it=1000,
    )
    spaces = sp.build_function_spaces(msh, config)
    source = sp._build_manual_line_source(msh, spaces, config)
    reverse_config = sp.PipelineConfig(
        nedelec_order=nedelec_order,
        source_start=config.source_end,
        source_end=config.source_start,
        source_mode="manual_line",
        source_quadrature_points=33,
        source_projection_mode="raw",
    )
    reverse = sp._build_manual_line_source(msh, spaces, reverse_config)
    reverse["vector"].scale(-1.0)

    relative = _relative_endpoint_residual(
        sp,
        msh,
        spaces["S"],
        spaces["V"],
        source["vector"],
        config,
    )
    reversal_error = _relative_vector_difference(source["vector"], reverse["vector"])

    assert spaces["S"].element.basix_element.degree == nedelec_order
    assert relative < 1.0e-9
    assert reversal_error < 1.0e-12
    diagnostics = source["local_projection_diagnostics"]
    assert diagnostics["integration_mode"] == "exact_tetra_intervals"
    assert diagnostics["interval_gate"]["passed"] is True
    assert diagnostics["quadrature_points_per_segment_min"] == nedelec_order + 1
    assert diagnostics["quadrature_points_per_segment_max"] == nedelec_order + 1
    source["vector"].destroy()
    reverse["vector"].destroy()


@pytest.mark.parametrize("nedelec_order", [1, 2])
def test_pec_source_gate_uses_free_s0_dofs_not_full_scalar_space(nedelec_order):
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    config = sp.PipelineConfig(
        nedelec_order=nedelec_order,
        source_start=(0.2, 0.285, 0.389),
        source_end=(0.8, 0.285, 0.389),
        source_mode="manual_line",
        source_quadrature_points=257,
        source_projection_mode="charge_conserving",
        outer_boundary_mode="pec",
        rtol=1.0e-11,
        atol=1.0e-13,
        max_it=1000,
    )
    spaces = sp.build_function_spaces(msh, config)
    source = sp.build_source(msh, spaces, config)
    assert sp._require_source_projection_gate(source)["passed"] is True
    facets = mesh.locate_entities_boundary(
        msh,
        msh.topology.dim - 1,
        lambda x: np.ones(x.shape[1], dtype=bool),
    )
    edge_dofs = np.unique(
        fem.locate_dofs_topological(spaces["V"], msh.topology.dim - 1, facets)
    )
    edge_global = spaces["V"].dofmap.index_map.local_to_global(edge_dofs).astype(np.int32)

    diagnostics = sp._validate_source_after_boundary_elimination(
        msh,
        spaces,
        source["vector"],
        config,
        edge_global,
    )

    assert diagnostics["constraint_space"] == "S0"
    assert diagnostics["boundary_scalar_dof_count"] > 0
    assert diagnostics["full_scalar_relative_residual"] > 0.05
    assert diagnostics["free_scalar_relative_residual"] < 1.0e-12
    assert diagnostics["relative_residual"] == pytest.approx(
        diagnostics["free_scalar_relative_residual"]
    )
    assert diagnostics["passed"] is True
    source["boundary_elimination_diagnostics"] = diagnostics
    assert sp._require_source_preflight_gates(source)["passed"] is True
    source["vector"].destroy()


def test_actual_source_after_pec_boundary_elimination_gates_bad_s0_gradient():
    pytest.importorskip("dolfinx.fem")
    pytest.importorskip("dolfinx.mesh")
    from dolfinx import fem, mesh
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
    config = sp.PipelineConfig(
        nedelec_order=2,
        source_start=(0.4, 0.4, 0.4),
        source_end=(0.6, 0.6, 0.6),
        source_mode="manual_line",
        source_quadrature_points=7,
        source_projection_mode="charge_conserving",
        outer_boundary_mode="pec",
    )
    spaces = sp.build_function_spaces(msh, config)
    source = sp.build_source(msh, spaces, config)
    facets = mesh.locate_entities_boundary(
        msh,
        msh.topology.dim - 1,
        lambda x: np.ones(x.shape[1], dtype=bool),
    )
    local_bc_dofs = np.unique(fem.locate_dofs_topological(spaces["V"], msh.topology.dim - 1, facets))
    bc_global = spaces["V"].dofmap.index_map.local_to_global(local_bc_dofs).astype(np.int32)

    diagnostics = sp._validate_source_after_boundary_elimination(
        msh,
        spaces,
        source["vector"],
        config,
        bc_global,
    )

    assert diagnostics["relative_residual"] < 1.0e-11
    assert diagnostics["relative_tolerance"] == pytest.approx(
        sp.SOURCE_PROJECTION_MAX_AFTER_RELATIVE_RESIDUAL
    )
    assert diagnostics["eliminated_l2_over_source"] < 1.0e-11

    bad_source = source["vector"].copy()
    boundary_scalar = np.unique(
        fem.locate_dofs_topological(spaces["S"], msh.topology.dim - 1, facets)
    )
    interior_scalar = min(
        set(range(spaces["S"].dofmap.index_map.size_global)).difference(
            int(value) for value in boundary_scalar
        )
    )
    scalar = fem.Function(spaces["S"])
    scalar.x.petsc_vec.setValue(interior_scalar, 1.0, addv=PETSc.InsertMode.INSERT_VALUES)
    scalar.x.petsc_vec.assemble()
    gradient = fem_petsc.discrete_gradient(spaces["S"], spaces["V"])
    gradient.assemble()
    perturbation = bad_source.duplicate()
    perturbation.set(0.0)
    gradient.mult(scalar.x.petsc_vec, perturbation)
    bad_source.axpy(1.0, perturbation)
    bad_source.assemble()
    bad_diagnostics = sp._validate_source_after_boundary_elimination(
        msh,
        spaces,
        bad_source,
        config,
        bc_global,
    )
    assert bad_diagnostics["passed"] is False
    bad_info = dict(source)
    bad_info["boundary_elimination_diagnostics"] = bad_diagnostics
    with pytest.raises(RuntimeError, match="source_boundary_elimination.relative_residual"):
        sp._require_source_preflight_gates(bad_info)
    bad_source.destroy()
    perturbation.destroy()
    gradient.destroy()
    source["vector"].destroy()
