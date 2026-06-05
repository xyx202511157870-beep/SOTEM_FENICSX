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


def test_scalar_potential_gauge_uses_single_owned_dof():
    sp = _load_pipeline_module()

    dofs = sp._scalar_potential_gauge_dofs(_FakeSpace())

    assert dofs.tolist() == [0]
