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
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_dc_secondary_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dc_secondary_zero_contrast_returns_near_zero_field():
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(ksp_type="cg", rtol=1.0e-10, atol=1.0e-12, max_it=200)
    spaces = sp.build_function_spaces(msh, config)
    sigma = fem.Function(spaces["Q"], name="sigma")
    sigma.x.array[:] = 0.01
    sigma.x.scatter_forward()
    materials = {"sigma": sigma, "sigma_initial": sigma}
    Ep0 = fem.Function(spaces["V"], name="Ep0")
    Ep0.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))
    Ep0.x.scatter_forward()

    result = sp._solve_dc_secondary_field(
        msh,
        spaces,
        materials,
        Ep0,
        config,
        sigma_background=0.01,
    )

    assert result["contrast_is_zero"] is True
    assert np.linalg.norm(result["Es0"].x.array) < 1.0e-10


def test_dc_secondary_nonzero_contrast_returns_finite_nonzero_field():
    from dolfinx import fem, mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(ksp_type="cg", rtol=1.0e-10, atol=1.0e-12, max_it=400)
    spaces = sp.build_function_spaces(msh, config)
    sigma = fem.Function(spaces["Q"], name="sigma")
    sigma.x.array[:] = np.linspace(0.01, 0.02, sigma.x.array.size)
    sigma.x.scatter_forward()
    materials = {"sigma": sigma, "sigma_initial": sigma}
    Ep0 = fem.Function(spaces["V"], name="Ep0")
    Ep0.interpolate(lambda x: np.vstack((np.ones(x.shape[1]), np.zeros(x.shape[1]), np.zeros(x.shape[1]))))
    Ep0.x.scatter_forward()

    result = sp._solve_dc_secondary_field(
        msh,
        spaces,
        materials,
        Ep0,
        config,
        sigma_background=0.01,
    )

    values = np.asarray(result["Es0"].x.array, dtype=float)
    assert result["contrast_is_zero"] is False
    assert np.all(np.isfinite(values))
    assert np.linalg.norm(values) > 0.0
    assert result["ksp_reason"] > 0
