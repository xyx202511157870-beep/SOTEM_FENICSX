from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from atem3d.seepage_channel_model import model_for_variant
import atem3d.seepage_channel_validation as validation
from atem3d.seepage_verification import (
    ModelContractMismatch,
    canonical_model_contract,
    model_fingerprint,
    require_consistent_fingerprints,
    verify_fenicsx_run_contract,
    verify_simpeg_config_contract,
)


def test_model_fingerprint_is_deterministic_and_geometry_sensitive() -> None:
    thin = model_for_variant("thin_60x1x1")
    baseline = model_for_variant("baseline_60x10x10")

    assert model_fingerprint(thin) == model_fingerprint(thin)
    assert model_fingerprint(thin) != model_fingerprint(baseline)
    assert canonical_model_contract(thin)["channel"]["size_m"] == [60.0, 1.0, 1.0]


def test_mixed_or_missing_model_fingerprints_fail_closed() -> None:
    thin_hash = model_fingerprint(model_for_variant("thin_60x1x1"))
    thick_hash = model_fingerprint(model_for_variant("baseline_60x10x10"))

    with pytest.raises(ModelContractMismatch, match="mixed model fingerprints"):
        require_consistent_fingerprints(
            {
                "simpeg_channel": {"model_fingerprint": thin_hash},
                "fenicsx_channel": {"model_fingerprint": thick_hash},
            }
        )
    with pytest.raises(ModelContractMismatch, match="missing model fingerprint"):
        require_consistent_fingerprints(
            {
                "simpeg_channel": {"model_fingerprint": thin_hash},
                "magnetic_audit": {},
            }
        )


def test_empymod_and_simpeg_payload_adapters_stamp_requested_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thin = model_for_variant("thin_60x1x1")
    values = np.ones((5, 31, 3), dtype=float)
    monkeypatch.setattr(validation, "run_empymod_reference", lambda *args, **kwargs: values)
    monkeypatch.setattr(validation, "load_simpeg_values", lambda *args, **kwargs: values)

    empymod = validation.empymod_background_payload(model=thin)
    simpeg = validation.simpeg_payload_from_h5("unused.h5", model=thin)

    assert str(np.asarray(empymod["model_fingerprint"]).item()) == model_fingerprint(thin)
    assert str(np.asarray(simpeg["model_fingerprint"]).item()) == model_fingerprint(thin)


def test_fenicsx_contract_is_derived_from_resolved_config_and_material_audit(
    tmp_path: Path,
) -> None:
    thin = model_for_variant("thin_60x1x1")
    (tmp_path / "run_config_resolved.yaml").write_text(
        "\n".join(
            (
                "source_start: [-50, 0, -0.1]",
                "source_end: [50, 0, -0.1]",
                "source_current: 1",
                "rho_air: 100000000",
                "rho_earth: 100",
                "t_min: 1e-05",
                "t_max: 0.01",
                "time_growth: 1.258925411794167",
                "magnetic_receiver_mode: biot_current",
                "magnetic_dbdt_mode: biot_rate",
            )
        ),
        encoding="utf-8",
    )
    summary = {
        "receiver_count": 5,
        "receiver_provenance": ["explicit_full_domain"] * 5,
        "material_audit": {
            "enabled": True,
            "bounds": [[-30.0, 30.0], [-0.5, 0.5], [-20.5, -19.5]],
            "sigma_s_per_m": 1.0,
            "theoretical_volume_m3": 60.0,
            "global_discrete_volume_m3": 60.01,
            "relative_volume_error": 1.0 / 6000.0,
        },
    }
    (tmp_path / "fenicsx_run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    provenance = verify_fenicsx_run_contract(tmp_path, model=thin, case="channel")

    assert provenance["model_fingerprint"] == model_fingerprint(thin)
    assert provenance["magnetic_dbdt_mode"] == "biot_rate"

    summary["material_audit"]["bounds"] = [
        [-30.0, 30.0],
        [-5.0, 5.0],
        [-25.0, -15.0],
    ]
    (tmp_path / "fenicsx_run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    with pytest.raises(ModelContractMismatch, match="conductivity-box bounds"):
        verify_fenicsx_run_contract(tmp_path, model=thin, case="channel")


def test_simpeg_thin_configs_match_the_same_model_contract() -> None:
    thin = model_for_variant("thin_60x1x1")

    for case in ("background", "channel"):
        provenance = verify_simpeg_config_contract(
            Path(f"examples/seepage_channel_100m_5rx_simpeg_thin_{case}.yaml"),
            model=thin,
            case=case,
        )
        assert provenance["model_fingerprint"] == model_fingerprint(thin)
        assert provenance["case"] == case
