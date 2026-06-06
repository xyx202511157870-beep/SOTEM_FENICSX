from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location("sotem_pipeline_for_total_field_ip_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_total_field_debye_step_metadata_records_task_book_equation():
    sp = _load_pipeline_module()
    debye = {
        "fit": SimpleNamespace(sigma_infinity=0.02),
        "terms": [
            sp.DebyeTerm(delta_sigma=0.003, tau=0.1),
            sp.DebyeTerm(delta_sigma=0.002, tau=1.0),
        ],
    }

    metadata = sp._debye_total_field_step_metadata(debye, dt=0.1)

    assert metadata["enabled"] is True
    assert metadata["time_scheme"] == "backward_euler"
    assert metadata["memory_initial_condition"] == "chi_k0 = E0"
    assert metadata["lhs_operator"] == "K + R + M(sigma_eff)/dt"
    assert metadata["rhs_history"] == "M[J_old + sum(delta_sigma_k * alpha_k * chi_old_k)]/dt"
    assert metadata["current_convention"] == "J = sigma_inf E - sum(delta_sigma_k chi_k)"
    assert metadata["sigma_eff"] == pytest.approx(0.02 - 0.003 * 0.5 - 0.002 / 11.0)
    assert metadata["alpha"] == pytest.approx([0.5, 10.0 / 11.0])
    assert metadata["beta"] == pytest.approx([0.5, 1.0 / 11.0])
    assert metadata["delta_sigma_zero_degenerates_to_noip"] is False


def test_total_field_debye_step_metadata_marks_zero_delta_noip_degeneracy():
    sp = _load_pipeline_module()
    debye = {
        "fit": SimpleNamespace(sigma_infinity=0.02),
        "terms": [sp.DebyeTerm(delta_sigma=0.0, tau=0.1)],
    }

    metadata = sp._debye_total_field_step_metadata(debye, dt=0.1)

    assert metadata["enabled"] is True
    assert metadata["sigma_eff"] == pytest.approx(0.02)
    assert metadata["sum_delta_sigma"] == pytest.approx(0.0)
    assert metadata["delta_sigma_zero_degenerates_to_noip"] is True


def test_total_field_debye_step_metadata_accepts_material_map_sigma_infinity():
    sp = _load_pipeline_module()
    debye = {
        "sigma_infinity": 0.02,
        "terms": [sp.DebyeTerm(delta_sigma=0.003, tau=0.1)],
    }

    metadata = sp._debye_total_field_step_metadata(debye, dt=0.1)

    assert metadata["sigma_inf"] == pytest.approx(0.02)
    assert metadata["sigma0"] == pytest.approx(0.017)
    assert metadata["sigma_eff"] == pytest.approx(0.0185)


def test_total_field_debye_step_metadata_reports_noip_when_no_terms():
    sp = _load_pipeline_module()

    metadata = sp._debye_total_field_step_metadata(None, dt=0.1)

    assert metadata == {
        "enabled": False,
        "time_scheme": "backward_euler",
        "reason": "no_debye_terms",
    }
