from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from atem3d.materials.material_map import CellMaterialMap, apply_leakage_channel_marker
from atem3d.materials.prony import DebyeTerm, PronyConductivity


pytest.importorskip("dolfinx.fem")
pytest.importorskip("dolfinx.mesh")


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_leakage_forward_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dolfinx_leakage_channel_material_map_runs_primary_secondary_forward():
    from atem3d.primary import ZeroPrimaryProvider
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 1)
    config = sp.PipelineConfig(
        receiver=(0.25, 0.25, 0.25),
        receiver_evaluation_mode="first_cell",
        outer_boundary_mode="natural",
        ksp_type="cg",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=400,
    )
    spaces = sp.build_function_spaces(msh, config)
    centers = sp._cell_centers(msh)
    background_marker = 1
    leakage_marker = 7
    markers = np.full(centers.shape[0], background_marker, dtype=int)
    markers = apply_leakage_channel_marker(
        markers,
        centers,
        channel_points=np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]]),
        radius=0.35,
        leakage_marker=leakage_marker,
    )
    material_map = CellMaterialMap(
        markers=markers,
        materials={
            background_marker: PronyConductivity.no_ip(0.01),
            leakage_marker: PronyConductivity(
                sigma_inf=0.05,
                terms=[DebyeTerm(delta_sigma=0.015, tau=0.1)],
            ),
        },
    )
    built_materials = sp._make_dolfinx_materials_from_cell_material_map(
        msh,
        spaces,
        material_map,
        dt=1.0e-5,
    )
    operators = sp.assemble_operators(
        msh,
        spaces,
        built_materials["materials"],
        facet_tags=None,
        config=config,
        debye=built_materials["debye"],
    )

    class ConstantPrimaryProvider(ZeroPrimaryProvider):
        def get_Ep_on_V(self, t, points):
            return np.column_stack((np.ones(len(points)), np.zeros(len(points)), np.zeros(len(points))))

        def get_Ep_dc_on_V(self, points):
            return np.column_stack((np.ones(len(points)), np.zeros(len(points)), np.zeros(len(points))))

        def get_receiver_E(self, t, receivers):
            return np.array([[10.0, 1.0, 0.0]])

        def get_receiver_dBdt(self, t, receivers):
            return np.zeros((1, 3))

    built_operator = sp._make_dolfinx_primary_secondary_forward_operator(
        msh,
        spaces,
        built_materials["materials"],
        operators,
        config,
        primary=ConstantPrimaryProvider(),
        receiver_locations=np.array([[0.25, 0.25, 0.25]]),
        components=("Ex", "Ey", "dBzdt"),
        material=built_materials["representative_material"],
        sigma_background=0.01,
        debye=built_materials["debye"],
    )

    predicted = built_operator["operator"].forward(np.array([1.0e-5]))

    assert predicted.shape == (1, 3)
    assert np.all(np.isfinite(predicted))
    assert built_materials["diagnostics"]["leakage_cell_count"] > 0
    assert built_materials["debye"] is not None
