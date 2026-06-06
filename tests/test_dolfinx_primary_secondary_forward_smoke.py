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
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_ps_forward_smoke", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dolfinx_primary_secondary_zero_contrast_forward_returns_primary_response():
    from atem3d.materials.prony import PronyConductivity
    from atem3d.primary import CachedPrimaryProvider
    from atem3d.solvers import PrimarySecondaryForwardOperator
    from dolfinx import mesh
    from mpi4py import MPI

    sp = _load_pipeline_module()
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    config = sp.PipelineConfig(
        receiver=(0.25, 0.25, 0.25),
        receiver_evaluation_mode="first_cell",
    )
    spaces = sp.build_function_spaces(msh, config)
    times = np.array([1.0e-5, 2.0e-5])
    fem_points = np.array([[0.25, 0.25, 0.25]])
    receiver_locations = np.array([[0.25, 0.25, 0.25]])
    primary = CachedPrimaryProvider(
        times=times,
        points=fem_points,
        receivers=receiver_locations,
        Ep_on_V=np.array([[[1.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]),
        receiver_E=np.array([[[10.0, 1.0, 0.0]], [[5.0, 0.5, 0.0]]]),
        receiver_dBdt=np.array([[[0.0, 0.0, -3.0]], [[0.0, 0.0, -1.5]]]),
        Ep_dc_on_V=np.array([[1.0, 0.0, 0.0]]),
    )
    operator = PrimarySecondaryForwardOperator(
        primary=primary,
        fem_points=fem_points,
        receiver_locations=receiver_locations,
        components=("Ex", "Ey", "dBzdt"),
        material=PronyConductivity.no_ip(0.01),
        sigma_background=0.01,
        secondary_receiver_projector=sp._make_dolfinx_zero_secondary_receiver_projector(
            msh,
            spaces,
            config,
        ),
    )

    predicted = operator.forward(times)

    np.testing.assert_allclose(predicted, [[10.0, 1.0, -3.0], [5.0, 0.5, -1.5]], atol=1.0e-12)
