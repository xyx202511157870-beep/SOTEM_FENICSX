from __future__ import annotations

import json
from pathlib import Path

import pytest
import numpy as np

from atem3d.seepage_channel_model import model_for_variant
import atem3d.seepage_channel_validation as validation
from atem3d.seepage_verification import (
    ModelContractMismatch,
    VerificationGateError,
    anomaly_energy_trend,
    build_verification_summary,
    canonical_model_contract,
    cross_solver_agreement,
    model_fingerprint,
    parity_metrics,
    require_consistent_fingerprints,
    three_level_convergence,
    verify_fenicsx_run_contract,
    verify_simpeg_config_contract,
    zero_contrast_metrics,
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
            "mesh_size_m": 0.25,
        },
    }
    (tmp_path / "fenicsx_run_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    provenance = verify_fenicsx_run_contract(
        tmp_path, model=thin, case="channel", expected_local_mesh_size=0.25
    )

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


def test_zero_contrast_metrics_exclude_center_and_normalize_by_background() -> None:
    background = np.ones((5, 4, 3), dtype=float)
    delta = np.zeros_like(background)
    delta[2, :, :] = 100.0
    delta[[0, 1, 3, 4], :, :] = 1.0e-7

    metrics = zero_contrast_metrics(delta, background, threshold=1.0e-6)

    assert metrics["pass"] is True
    assert metrics["formal_receiver_indices"] == [0, 1, 3, 4]
    assert all(value == pytest.approx(1.0e-7) for value in metrics["normalized_l2"])


def test_anomaly_energy_trend_is_monotone_from_zero_contrast() -> None:
    background = np.ones((5, 4, 3), dtype=float)
    deltas = [np.full_like(background, value) for value in (0.0, 0.01, 0.1, 0.5)]

    summary = anomaly_energy_trend(
        [0.01, 0.02, 0.1, 1.0], deltas, background
    )

    assert summary["pass"] is True
    assert summary["energies"] == sorted(summary["energies"])


def test_three_level_convergence_requires_decreasing_error_and_thresholds() -> None:
    fine = np.ones((5, 4, 3), dtype=float)
    medium = fine * 1.04
    coarse = fine * 1.20

    summary = three_level_convergence(
        coarse,
        medium,
        fine,
        median_threshold=0.10,
        p95_threshold=0.20,
    )

    assert summary["pass"] is True
    assert summary["fine_medium"]["median"] < summary["medium_coarse"]["median"]


def test_parity_metrics_apply_even_ex_and_odd_magnetic_contracts() -> None:
    values = np.zeros((5, 3, 3), dtype=float)
    values[0, :, 0], values[4, :, 0] = 2.0, 2.0
    values[1, :, 0], values[3, :, 0] = 1.0, 1.0
    values[2, :, 0] = 3.0
    for component in (1, 2):
        values[0, :, component], values[4, :, component] = 2.0, -2.0
        values[1, :, component], values[3, :, component] = 1.0, -1.0
        values[2, :, component] = 0.0

    summary = parity_metrics(values, pair_threshold=0.05, center_threshold=0.02)

    assert summary["pass"] is True
    assert summary["components"]["dBzdt"]["parity"] == "odd"
    assert summary["components"]["Hz"]["center_ratio"] == 0.0


def test_cross_solver_agreement_uses_formal_strong_signal_percentiles() -> None:
    first = np.ones((5, 6, 3), dtype=float)
    second = first * 1.10
    second[2, :, :] = 1000.0

    summary = cross_solver_agreement(
        first,
        second,
        median_threshold=0.20,
        p95_threshold=0.35,
    )

    assert summary["pass"] is True
    assert all(item["median"] < 0.20 for item in summary["components"].values())


def test_verification_summary_fails_closed_on_missing_or_failed_gate() -> None:
    fingerprint = model_fingerprint(model_for_variant("thin_60x1x1"))
    required = ("zero_contrast", "spatial_convergence", "cross_solver")

    missing = build_verification_summary(
        model_fingerprint_value=fingerprint,
        required_gates=required,
        gates={"zero_contrast": {"available": True, "pass": True}},
    )
    assert missing["pass"] is False
    assert missing["gates"]["spatial_convergence"]["available"] is False

    failed = build_verification_summary(
        model_fingerprint_value=fingerprint,
        required_gates=required,
        gates={name: {"available": True, "pass": name != "cross_solver"} for name in required},
    )
    assert failed["pass"] is False
    with pytest.raises(VerificationGateError, match="cross_solver"):
        build_verification_summary(
            model_fingerprint_value=fingerprint,
            required_gates=required,
            gates=failed["gates"],
            require_pass=True,
        )
