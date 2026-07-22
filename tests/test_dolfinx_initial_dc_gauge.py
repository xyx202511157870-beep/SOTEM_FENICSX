from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_dc_gauge_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeIndexMap:
    local_range = (0, 8)


class _FakeDofMap:
    index_map = _FakeIndexMap()


class _FakeSpace:
    dofmap = _FakeDofMap()


class _FakePC:
    def __init__(self):
        self.pc_type = None
        self.hypre_type = None

    def setType(self, value):
        self.pc_type = value

    def setHYPREType(self, value):
        self.hypre_type = value


class _FakeKSP:
    def __init__(self):
        self.ksp_type = None
        self.tolerances = None
        self.pc = _FakePC()
        self.set_from_options_called = False

    def setType(self, value):
        self.ksp_type = value

    def setTolerances(self, **values):
        self.tolerances = values

    def getPC(self):
        return self.pc

    def setFromOptions(self):
        self.set_from_options_called = True


def test_scalar_potential_gauge_uses_single_owned_dof():
    sp = _load_pipeline_module()

    dofs = sp._scalar_potential_gauge_dofs(_FakeSpace())

    assert dofs.tolist() == [0]


def test_initial_dc_ksp_uses_pipeline_solver_instead_of_hardcoded_cg():
    sp = _load_pipeline_module()
    ksp = _FakeKSP()
    config = sp.PipelineConfig(ksp_type="gmres", rtol=2.0e-9, atol=3.0e-13, max_it=77)

    sp._configure_initial_dc_ksp(ksp, config)

    assert ksp.ksp_type == "gmres"
    assert ksp.tolerances == {"rtol": 2.0e-9, "atol": 3.0e-13, "max_it": 1000}
    assert ksp.pc.pc_type == "hypre"
    assert ksp.pc.hypre_type == "boomeramg"
    assert ksp.set_from_options_called is True
