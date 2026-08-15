from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_pipeline_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "dolfinx" / "sotem_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "sotem_pipeline_for_magnetic6_production_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_magnetic6_contract_is_opt_in_and_fixed_order():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig()

    assert config.magnetic_output_contract == "legacy"
    assert sp.MAGNETIC6_COMPONENTS == (
        "Hx",
        "Hy",
        "Hz",
        "dBxdt",
        "dBydt",
        "dBzdt",
    )
    assert sp.MAGNETIC6_UNITS == (
        "A/m",
        "A/m",
        "A/m",
        "T/s",
        "T/s",
        "T/s",
    )


def test_e_form_magnetic6_requires_total_biot_h_and_curl_dbdt():
    sp = _load_pipeline_module()
    accepted = sp.PipelineConfig(
        formulation="e",
        magnetic_receiver_mode="biot_current",
        magnetic_dbdt_mode="curl",
        magnetic_output_contract="magnetic6",
    )

    assert tuple(sp._forward_components(accepted)) == sp.MAGNETIC6_COMPONENTS

    with pytest.raises(ValueError, match="biot_current"):
        sp._forward_components(
            sp.PipelineConfig(
                formulation="e",
                magnetic_receiver_mode="faraday_integrated",
                magnetic_dbdt_mode="curl",
                magnetic_output_contract="magnetic6",
            )
        )

    with pytest.raises(ValueError, match="magnetic_dbdt_mode='curl'"):
        sp._forward_components(
            sp.PipelineConfig(
                formulation="e",
                magnetic_receiver_mode="biot_current",
                magnetic_dbdt_mode="biot_rate",
                magnetic_output_contract="magnetic6",
            )
        )


def test_h_form_exposes_native_h_three_components():
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        formulation="h",
        magnetic_receiver_mode="faraday_integrated",
        magnetic_dbdt_mode="curl",
        magnetic_output_contract="magnetic6",
    )

    assert tuple(sp._forward_components(config)) == sp.MAGNETIC6_COMPONENTS


def test_biot_assignment_preserves_all_h_components():
    sp = _load_pipeline_module()
    record = {"dBxdt": 1.0, "dBydt": 2.0, "dBzdt": 3.0}

    sp._assign_biot_receiver_hz(record, np.array([4.0, 5.0, 6.0]))

    assert record == {
        "dBxdt": 1.0,
        "dBydt": 2.0,
        "dBzdt": 3.0,
        "Hx": 4.0,
        "Hy": 5.0,
        "Hz": 6.0,
    }


def test_magnetic6_writer_emits_canonical_npz(tmp_path: Path):
    sp = _load_pipeline_module()
    config = sp.PipelineConfig(
        workdir=tmp_path,
        receiver=(1.0, 2.0, 3.0),
        formulation="e",
        magnetic_receiver_mode="biot_current",
        magnetic_dbdt_mode="curl",
        magnetic_output_contract="magnetic6",
        ramp_off_time=5.0e-6,
    )
    times = np.array([1.0e-6, 2.0e-6, 5.0e-6])
    data = np.arange(18.0).reshape(3, 6)

    path = sp._write_magnetic6_numerical_npz(
        config,
        {
            "times": times,
            "data": data,
            "components": list(sp.MAGNETIC6_COMPONENTS),
        },
    )

    assert path == tmp_path / "magnetic6_numerical.npz"
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["times"], times)
        np.testing.assert_allclose(archive["data"], data[:, None, :])
        assert tuple(archive["components"].astype(str)) == sp.MAGNETIC6_COMPONENTS
        assert tuple(archive["units"].astype(str)) == sp.MAGNETIC6_UNITS
        np.testing.assert_allclose(archive["receiver_locations"], [[1.0, 2.0, 3.0]])
        assert str(archive["coordinate_system"].item()) == "z_up"
        assert str(archive["time_origin"].item()) == "after_ramp"
        assert float(archive["ramp_off_time"].item()) == pytest.approx(5.0e-6)
        assert int(archive["nedelec_order"].item()) == 2
        assert str(archive["magnetic_output_contract"].item()) == "magnetic6"
