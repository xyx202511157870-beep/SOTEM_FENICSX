import numpy as np
from pathlib import Path
import json
import pytest
import atem3d.seepage_channel_validation as validation

from atem3d.seepage_channel_model import MODEL, benchmark_model
from atem3d.seepage_channel_validation import (
    aggregate_payloads,
    channel_delta,
    strong_signal_mask,
    summarize_convergence,
)
from atem3d.seepage_verification import ModelContractMismatch, model_fingerprint


def test_channel_delta_preserves_sign() -> None:
    background = np.array([[[2.0, -3.0, 4.0]]])
    channel = np.array([[[1.0, -1.0, 8.0]]])
    np.testing.assert_allclose(
        channel_delta(channel, background),
        [[[-1.0, 2.0, 4.0]]],
    )


def test_strong_signal_mask_uses_component_peak_fraction() -> None:
    values = np.array([1.0, 0.04, 0.0, -0.2])
    np.testing.assert_array_equal(
        strong_signal_mask(values, 0.05),
        [True, False, False, True],
    )


def test_convergence_median_is_reported_without_dropping_raw_values() -> None:
    coarse = np.array([1.0, 2.0, 0.0])
    refined = np.array([1.02, 1.90, 0.01])
    summary = summarize_convergence(
        coarse,
        refined,
        strong_mask=np.array([True, True, False]),
    )
    assert summary["raw_count"] == 3
    assert summary["strong_count"] == 2
    assert summary["median_relative_change"] > 0.0


def test_aggregate_writes_canonical_artifacts_without_channel_empymod_error(
    tmp_path: Path,
) -> None:
    times = np.logspace(-5, -2, 31)
    locations = np.asarray([(0.0, y, -0.1) for y in (-20, -10, 0, 10, 20)])

    def payload(value: float, *, empymod: bool = False):
        result = {
            "times": times,
            "receiver_locations": locations,
            "components": np.asarray(("Ex", "dBzdt", "Hz")),
            "values": np.full((5, 31, 3), value),
            "model_fingerprint": model_fingerprint(MODEL),
        }
        if empymod:
            result["background_only_1d"] = True
        return result

    aggregate_payloads(
        tmp_path,
        empymod_background=payload(1.0, empymod=True),
        simpeg_background=payload(1.1),
        simpeg_channel=payload(1.3),
        fenicsx_background=payload(0.9),
        fenicsx_channel=payload(1.25),
    )
    required = {
        "benchmark_values.csv",
        "background_errors.csv",
        "channel_delta_values.csv",
        "channel_delta_errors.csv",
        "model_audit.json",
        "convergence_summary.json",
        "benchmark_summary.json",
        "benchmark_results.npz",
    }
    assert required.issubset(path.name for path in tmp_path.iterdir())
    background_error_text = (tmp_path / "background_errors.csv").read_text(
        encoding="utf-8"
    )
    assert "channel" not in background_error_text.lower()


def test_aggregate_writes_thin_variant_model_audit(tmp_path: Path) -> None:
    model = benchmark_model("thin_60x1x1")

    def payload(value: float, *, empymod: bool = False):
        result = {
            "times": model.times,
            "receiver_locations": np.asarray(model.receiver_locations),
            "components": np.asarray(("Ex", "dBzdt", "Hz")),
            "values": np.full((5, 31, 3), value),
            "model_fingerprint": model_fingerprint(model),
        }
        if empymod:
            result["background_only_1d"] = True
        return result

    aggregate_payloads(
        tmp_path,
        empymod_background=payload(1.0, empymod=True),
        simpeg_background=payload(1.1),
        simpeg_channel=payload(1.2),
        fenicsx_background=payload(0.9),
        fenicsx_channel=payload(1.15),
        model=model,
        variant="thin_60x1x1",
    )
    audit = json.loads((tmp_path / "model_audit.json").read_text(encoding="utf-8"))
    assert audit["variant"] == "thin_60x1x1"
    assert audit["channel"]["size_m"] == [60.0, 1.0, 1.0]
    assert audit["channel"]["theoretical_volume_m3"] == 60.0
    assert audit["model_fingerprint"] == model_fingerprint(model)


def test_aggregate_rejects_payload_from_a_different_model(tmp_path: Path) -> None:
    thin = benchmark_model("thin_60x1x1")
    thin_hash = model_fingerprint(thin)
    thick_hash = model_fingerprint(MODEL)

    def payload(value: float, fingerprint: str, *, empymod: bool = False):
        result = {
            "times": thin.times,
            "receiver_locations": np.asarray(thin.receiver_locations),
            "components": np.asarray(("Ex", "dBzdt", "Hz")),
            "values": np.full((5, 31, 3), value),
            "model_fingerprint": fingerprint,
        }
        if empymod:
            result["background_only_1d"] = True
        return result

    with pytest.raises(ModelContractMismatch, match="mixed model fingerprints"):
        aggregate_payloads(
            tmp_path,
            empymod_background=payload(1.0, thin_hash, empymod=True),
            simpeg_background=payload(1.1, thin_hash),
            simpeg_channel=payload(1.2, thin_hash),
            fenicsx_background=payload(0.9, thin_hash),
            fenicsx_channel=payload(1.15, thick_hash),
            model=thin,
            variant="thin_60x1x1",
        )


def test_directory_aggregation_verifies_both_fenicsx_run_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    thin = benchmark_model("thin_60x1x1")
    fingerprint = model_fingerprint(thin)
    payload = {
        "times": thin.times,
        "receiver_locations": np.asarray(thin.receiver_locations),
        "components": np.asarray(("Ex", "dBzdt", "Hz")),
        "values": np.ones((5, 31, 3)),
        "model_fingerprint": np.asarray(fingerprint),
    }
    for name in ("empymod_background", "simpeg_background", "simpeg_channel"):
        np.savez_compressed(tmp_path / f"{name}.npz", **payload)

    calls: list[str] = []

    def verify(run_dir, *, model, case):
        calls.append(case)
        return {"model_fingerprint": model_fingerprint(model), "case": case}

    monkeypatch.setattr(validation, "verify_fenicsx_run_contract", verify, raising=False)
    monkeypatch.setattr(
        validation,
        "fenicsx_payload_from_csv",
        lambda path, model=thin: dict(payload),
    )
    monkeypatch.setattr(
        validation,
        "aggregate_payloads",
        lambda *args, **kwargs: {"model_fingerprint": fingerprint},
    )

    validation.aggregate_result_directory(
        tmp_path, model=thin, variant="thin_60x1x1"
    )

    assert calls == ["background", "channel"]
    for case in calls:
        provenance_path = tmp_path / f"fenicsx_{case}_provenance.json"
        assert json.loads(provenance_path.read_text(encoding="utf-8"))[
            "model_fingerprint"
        ] == fingerprint
