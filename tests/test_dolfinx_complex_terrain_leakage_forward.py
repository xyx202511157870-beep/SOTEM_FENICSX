from __future__ import annotations

import importlib.util
import json
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


def test_gmsh_terrain_leakage_mesh_runs_primary_secondary_forward(tmp_path):
    pytest.importorskip("gmsh")

    from atem3d.primary import ZeroPrimaryProvider

    sp = _load_pipeline_module()
    mesh_path = tmp_path / "terrain_leakage.msh"
    mesh_info = sp._write_small_gmsh_terrain_leakage_mesh(mesh_path, mesh_size=0.65)
    config = sp.PipelineConfig(
        workdir=tmp_path,
        msh_name=mesh_path.name,
        receiver=(0.25, 0.25, -0.25),
        receiver_evaluation_mode="first_cell",
        outer_boundary_mode="natural",
        ksp_type="cg",
        rtol=1.0e-10,
        atol=1.0e-12,
        max_it=400,
    )
    msh, _cell_tags, facet_tags = sp.load_mesh(config)
    spaces = sp.build_function_spaces(msh, config)
    centers = sp._cell_centers(msh)
    background_marker = 1
    leakage_marker = 7
    markers = np.full(centers.shape[0], background_marker, dtype=int)
    markers = apply_leakage_channel_marker(
        markers,
        centers,
        channel_points=np.array([[0.05, 0.5, -0.45], [0.95, 0.5, -0.45]]),
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
        facet_tags=facet_tags,
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
        receiver_locations=np.array([[0.25, 0.25, -0.25]]),
        components=("Ex", "Ey", "dBzdt"),
        material=built_materials["representative_material"],
        sigma_background=0.01,
        debye=built_materials["debye"],
    )

    predicted = built_operator["operator"].forward(np.array([1.0e-5]))

    assert predicted.shape == (1, 3)
    assert np.all(np.isfinite(predicted))
    assert mesh_info["terrain_elevation_max"] > mesh_info["terrain_elevation_min"]
    assert built_materials["diagnostics"]["leakage_cell_count"] > 0


def test_corrected_leakage_convergence_runner_writes_dolfinx_refined_artifacts(tmp_path):
    from atem3d.corrected_model import (
        CorrectedModelValidationConfig,
        build_corrected_leakage_channel_case_specs,
    )
    from atem3d.corrected_model_runner import run_corrected_model_convergence_validation

    config = CorrectedModelValidationConfig(n_observation_times=2)
    spec = build_corrected_leakage_channel_case_specs(tmp_path, config=config)["noip"]
    spec["dolfinx_forward"]["cells"] = [1, 1, 1]
    spec["convergence_reference"]["dolfinx_forward"]["cells"] = [2, 1, 1]
    spec["output_dir"] = str(tmp_path / "noip_convergence")

    summary = run_corrected_model_convergence_validation(spec)

    assert summary["reference_type"] == "dolfinx_refined"
    assert summary["final_acceptance_passed"] is False
    assert (tmp_path / "noip_convergence" / "predictions.csv").is_file()
    assert (tmp_path / "noip_convergence" / "reference_empymod_or_1d.csv").is_file()
    assert (tmp_path / "noip_convergence" / "diagnostics.json").is_file()
    diagnostics = json.loads((tmp_path / "noip_convergence" / "diagnostics.json").read_text(encoding="utf-8"))
    secondary = diagnostics["secondary_effect_diagnostic"]
    assert secondary["reference_type"] == "dolfinx_refined"
    assert secondary["component_names"] == ["Ex", "Ey", "dBzdt"]
    assert secondary["prediction_secondary_effect_nonzero"] is True
    assert secondary["reference_secondary_effect_nonzero"] is True
    assert secondary["secondary_effect_nonzero"] is True


def test_corrected_runner_gmsh_terrain_mesh_preflight_marks_leakage_cells(tmp_path):
    pytest.importorskip("gmsh")

    from atem3d.corrected_model_runner import (
        _build_dolfinx_forward_mesh,
        _terrain_mesh_runtime_config,
    )
    from atem3d.materials.material_map import apply_leakage_channel_marker_with_diagnostics
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    forward_cfg = {
        "terrain_mesh": {
            "mode": "small_gmsh_terrain_leakage",
            "mesh_size": 0.9,
            "msh_name": "terrain_leakage.msh",
        }
    }
    terrain_runtime = _terrain_mesh_runtime_config(forward_cfg, output_dir=tmp_path)
    config = sp.PipelineConfig(
        workdir=tmp_path,
        msh_name="terrain_leakage.msh",
        receiver=(0.25, 0.25, -0.25),
        receiver_evaluation_mode="first_cell",
        outer_boundary_mode="natural",
    )

    msh, facet_tags, mesh_runtime = _build_dolfinx_forward_mesh(
        sp,
        mesh,
        MPI,
        config,
        domain_min=np.array([0.0, 0.0, -1.0]),
        domain_max=np.array([1.0, 1.0, 0.2]),
        cells=[1, 1, 1],
        terrain_runtime=terrain_runtime,
    )

    centers = sp._cell_centers(msh)
    marker_result = apply_leakage_channel_marker_with_diagnostics(
        np.ones(centers.shape[0], dtype=int),
        centers,
        channel_points=np.array([[0.05, 0.5, -0.45], [0.95, 0.5, -0.45]]),
        radius=0.35,
        leakage_marker=7,
        min_marked_cells=1,
    )

    assert facet_tags is not None
    assert marker_result.diagnostics["leakage_cell_count"] > 0
    terrain = mesh_runtime["terrain_mesh"]
    assert terrain["mode"] == "small_gmsh_terrain_leakage"
    assert terrain["terrain_elevation_max"] > terrain["terrain_elevation_min"]
    assert Path(terrain["mesh_path"]).is_file()
