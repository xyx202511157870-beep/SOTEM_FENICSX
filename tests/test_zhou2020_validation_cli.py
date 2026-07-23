from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from atem3d.zhou2020_validation_cli import (
    compare_registered,
    finalize_run,
    prepare_run,
    publish_reference,
    record_convergence,
    register_fenicsx,
    verify_run,
)


TIMES = np.geomspace(1.0e-4, 3.0, 101)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _responses() -> tuple[np.ndarray, np.ndarray]:
    log_time = np.log10(TIMES)
    noip = np.column_stack(
        (
            np.exp(-TIMES / 0.3),
            2.0 * np.exp(-TIMES / 0.7),
            -3.0 * np.exp(-TIMES / 0.2),
        )
    )
    ip = noip.copy()
    ip[:, 0] = noip[:, 0] - 0.55 / (1.0 + np.exp(-(log_time + 1.2) * 8.0))
    ip[:, 1] = noip[:, 1] * (1.0 + 0.002 * np.exp(-TIMES))
    ip[:, 2] = noip[:, 2] * (1.0 - 0.01 * np.exp(-TIMES))
    return noip, ip


def _write_reference(path: Path, *, complete: bool = True) -> None:
    path.mkdir()
    noip, ip = _responses()
    for variant, values in (("noip", noip), ("ip", ip)):
        csv_path = path / f"empymod_{variant}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("time_s", "Ex_V_per_m", "Hz_A_per_m", "dBzdt_T_per_s")
            )
            writer.writerows(np.column_stack((TIMES, values)))
    (path / "empymod_srcpts_convergence.json").write_text(
        json.dumps({"status": "passed"}), encoding="utf-8"
    )
    artifacts = {}
    for name in (
        "empymod_noip.csv",
        "empymod_ip.csv",
        "empymod_srcpts_convergence.json",
    ):
        artifacts[name] = {"path": name, "sha256": _sha256(path / name)}
    (path / "reference_manifest.json").write_text(
        json.dumps(
            {
                "schema": "atem3d.zhou2020.reference-manifest/v1",
                "status": "reference_verified" if complete else "failed",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )


def _write_fenicsx(
    path: Path,
    *,
    variant: str,
    time_values: np.ndarray = TIMES,
    perturbation: float = 0.0,
) -> None:
    path.mkdir()
    noip, ip = _responses()
    values = noip if variant == "noip" else ip
    values = values[: time_values.size].copy()
    if perturbation:
        values[50, 0] += perturbation
    with (path / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_obs", "Ex", "Ey", "Hz", "dBzdt"))
        writer.writerows(
            np.column_stack((time_values, values[:, 0], 0.0 * time_values, values[:, 1:]))
        )
    (path / "run_config_resolved.yaml").write_text(
        f"variant: {variant}\n", encoding="utf-8"
    )


@pytest.fixture
def snapshots(tmp_path: Path) -> tuple[Path, Path]:
    case = tmp_path / "case.yaml"
    provenance = tmp_path / "provenance.json"
    case.write_text("case_id: zhou2020_grounded_wire\n", encoding="utf-8")
    provenance.write_text('{"case_id":"zhou2020_grounded_wire"}\n', encoding="utf-8")
    return case, provenance


def _prepared(tmp_path: Path, snapshots: tuple[Path, Path]) -> Path:
    return prepare_run(
        tmp_path / "runs",
        case_path=snapshots[0],
        provenance_path=snapshots[1],
        run_id="20260723T120000Z_test",
    )


def test_prepare_snapshots_inputs_and_rejects_unsafe_run_id(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"
    assert manifest["snapshots"]["case"]["sha256"] == _sha256(
        run_dir / "snapshots" / "case.yaml"
    )
    with pytest.raises(ValueError, match="run_id"):
        prepare_run(
            tmp_path / "other",
            case_path=snapshots[0],
            provenance_path=snapshots[1],
            run_id="../escape",
        )


def test_publish_reference_is_fail_closed(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    source = tmp_path / "reference"
    _write_reference(source, complete=False)
    with pytest.raises(ValueError, match="reference_verified"):
        publish_reference(run_dir, source)
    assert not (run_dir / "reference").exists()


def test_ip_registration_requires_passed_noip_gate(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    reference = tmp_path / "reference"
    _write_reference(reference)
    publish_reference(run_dir, reference)
    ip = tmp_path / "ip"
    _write_fenicsx(ip, variant="ip")
    with pytest.raises(RuntimeError, match="no-IP gate"):
        register_fenicsx(run_dir, variant="ip", level="S1T1B1", source_dir=ip)


def test_partial_window_cannot_pass_noip_gate(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    reference = tmp_path / "reference"
    _write_reference(reference)
    publish_reference(run_dir, reference)
    noip = tmp_path / "noip"
    _write_fenicsx(noip, variant="noip", time_values=TIMES[:-1])
    register_fenicsx(run_dir, variant="noip", level="S1T1B1", source_dir=noip)
    result = compare_registered(run_dir, level="S1T1B1", mode="noip")
    assert result["status"] == "incomplete_time_window"
    assert not result["passed"]


def test_exact_full_sequence_can_be_finalized_and_verified(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    reference = tmp_path / "reference"
    _write_reference(reference)
    publish_reference(run_dir, reference)

    noip = tmp_path / "noip"
    _write_fenicsx(noip, variant="noip")
    register_fenicsx(run_dir, variant="noip", level="S1T1B1", source_dir=noip)
    assert compare_registered(run_dir, level="S1T1B1", mode="noip")["passed"]

    ip = tmp_path / "ip"
    _write_fenicsx(ip, variant="ip")
    register_fenicsx(run_dir, variant="ip", level="S1T1B1", source_dir=ip)
    comparison = compare_registered(run_dir, level="S1T1B1", mode="full")
    assert comparison["status"] == "ip_internally_validated"

    convergence = tmp_path / "convergence.json"
    convergence.write_text(
        json.dumps({"status": "passed", "all_gates_passed": True}),
        encoding="utf-8",
    )
    record_convergence(run_dir, convergence)
    final = finalize_run(run_dir, level="S1T1B1")
    assert final["status"] == "ip_internally_validated"
    assert verify_run(run_dir)["verified"]


def test_failed_full_comparison_is_preserved(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    reference = tmp_path / "reference"
    _write_reference(reference)
    publish_reference(run_dir, reference)
    noip = tmp_path / "noip"
    _write_fenicsx(noip, variant="noip")
    register_fenicsx(run_dir, variant="noip", level="S1T1B1", source_dir=noip)
    compare_registered(run_dir, level="S1T1B1", mode="noip")
    ip = tmp_path / "ip"
    _write_fenicsx(ip, variant="ip", perturbation=100.0)
    register_fenicsx(run_dir, variant="ip", level="S1T1B1", source_dir=ip)
    result = compare_registered(run_dir, level="S1T1B1", mode="full")
    assert result["status"] == "failed_with_reproducible_evidence"
    assert (run_dir / "comparisons" / "S1T1B1" / "strict_comparison.json").exists()
    with pytest.raises(RuntimeError, match="strict comparison"):
        finalize_run(run_dir, level="S1T1B1")


def test_tampering_is_detected(
    tmp_path: Path, snapshots: tuple[Path, Path]
) -> None:
    run_dir = _prepared(tmp_path, snapshots)
    reference = tmp_path / "reference"
    _write_reference(reference)
    publish_reference(run_dir, reference)
    (run_dir / "reference" / "empymod_noip.csv").write_text(
        "tampered\n", encoding="utf-8"
    )
    result = verify_run(run_dir)
    assert not result["verified"]
    assert result["hash_failures"]
