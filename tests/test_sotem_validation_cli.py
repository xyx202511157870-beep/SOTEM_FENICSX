from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

import atem3d.sotem_validation_cli as cli
from atem3d.sotem_simpeg_adapter import build_benchmark_config


ROOT = Path(__file__).resolve().parents[1]
LEI_CASE = ROOT / "benchmarks" / "sotem" / "lei2023_noip.yaml"
SONG_CASE = ROOT / "benchmarks" / "sotem" / "song2025_layered_pair.yaml"
SIMPEG_SOLVER = "atem3d_simpeg_discretize_debye"


def _prepare(
    tmp_path,
    case=LEI_CASE,
    *,
    solver="empymod",
    level="S0T0B0",
    run_name="run",
):
    run_dir = tmp_path / run_name
    assert cli.main(
        [
            "prepare",
            "--case",
            str(case),
            "--solver",
            solver,
            "--level",
            level,
            "--run-dir",
            str(run_dir),
        ]
    ) == 0
    return run_dir


def _command(name, run_dir, case, *, resume=True, extra=()):
    args = [name, "--run-dir", str(run_dir), "--case", str(case)]
    if resume:
        args.append("--resume")
    args.extend(extra)
    return args


def _fake_response(case, scale=1.0):
    times = np.asarray(case.observation_times, dtype=float)
    base = np.arange(1, times.size + 1, dtype=float) * scale
    return {
        "times": times,
        "data": np.column_stack((base, -base, 0.5 * base, -0.25 * base)),
        "components": ["Ex", "Ey", "Hz", "dBzdt"],
    }


def _valid_ip_material_fit():
    case = cli.load_benchmark_case(SONG_CASE)
    config = build_benchmark_config(
        case,
        variant="ip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    return config["adapter_metadata"]["material_fit"]


def _valid_simpeg_provenance(case, variant, level="S0T0B0"):
    substeps = {"T0": 1, "T1": 2, "T2": 4}[level[2:4]]
    config = build_benchmark_config(
        case,
        variant=variant,
        spatial_level=level[0:2],
        boundary_level=level[4:6],
        substeps=substeps,
    )
    mesh = config["mesh"]["metadata"]
    return {
        "mesh_hash": config["mesh_hash"],
        "time_hash": config["time_hash"],
        "mesh_stats": {
            "n_cells": mesh["n_cells"],
            "n_edges": mesh["n_edges"],
            "axis_cell_counts": mesh["axis_cell_counts"],
            "bounds_m": mesh["public_bounds_z_down_m"],
            "internal_bounds_z_up_m": mesh["bounds_m"],
            "spatial_level": level[0:2],
            "boundary_level": level[4:6],
            "mesh_hash": config["mesh_hash"],
        },
        "coordinate_system": "z_down",
        "coordinate_transform": config["adapter_metadata"]["coordinate_transform"],
    }


def _build_effect_source_runs(
    tmp_path, monkeypatch, *, level="S0T0B0", case_path=SONG_CASE
):
    case = cli.load_benchmark_case(case_path)

    def fake_reference(times, config, *, mode):
        scale = 1.0 if mode == "noip" else 1.5
        return {**_fake_response(case, scale=scale), "reference_mode": mode}

    def fake_simpeg(case_value, **kwargs):
        scale = 1.1 if kwargs["variant"] == "noip" else 1.6
        return {
            **_fake_response(case_value, scale=scale),
            "solver_id": SIMPEG_SOLVER,
            **_valid_simpeg_provenance(case_value, kwargs["variant"], level),
            "variant": kwargs["variant"],
            "material_fit": (
                _valid_ip_material_fit() if kwargs["variant"] == "ip" else None
            ),
        }

    monkeypatch.setattr(cli, "get_empymod_reference", fake_reference)
    monkeypatch.setattr(cli, "run_simpeg_benchmark", fake_simpeg)
    runs = {
        "noip_reference": _prepare(
            tmp_path,
            case=case_path,
            solver="empymod",
            level=level,
            run_name="noip-reference",
        ),
        "ip_reference": _prepare(
            tmp_path,
            case=case_path,
            solver="empymod",
            level=level,
            run_name="ip-reference",
        ),
        "noip_simpeg": _prepare(
            tmp_path,
            case=case_path,
            solver=SIMPEG_SOLVER,
            level=level,
            run_name="noip-simpeg",
        ),
        "ip_simpeg": _prepare(
            tmp_path,
            case=case_path,
            solver=SIMPEG_SOLVER,
            level=level,
            run_name="ip-simpeg",
        ),
    }
    assert cli.main(
        _command(
            "reference",
            runs["noip_reference"],
            case_path,
            extra=("--variant", "noip"),
        )
    ) == 0
    assert cli.main(
        _command(
            "reference",
            runs["ip_reference"],
            case_path,
            extra=("--variant", "cole-cole-exact"),
        )
    ) == 0
    assert cli.main(
        _command(
            "simpeg",
            runs["noip_simpeg"],
            case_path,
            extra=("--variant", "noip"),
        )
    ) == 0
    assert cli.main(
        _command(
            "simpeg",
            runs["ip_simpeg"],
            case_path,
            extra=("--variant", "ip"),
        )
    ) == 0
    return runs


def _effect_args(effect_run, runs, *, case=SONG_CASE):
    return _command(
        "effect",
        effect_run,
        case,
        extra=(
            "--noip-simpeg-run",
            str(runs["noip_simpeg"]),
            "--noip-reference-run",
            str(runs["noip_reference"]),
            "--ip-simpeg-run",
            str(runs["ip_simpeg"]),
            "--ip-reference-run",
            str(runs["ip_reference"]),
        ),
    )


def test_prepare_writes_unique_complete_manifests(tmp_path):
    args = [
        "prepare",
        "--case",
        str(LEI_CASE),
        "--solver",
        "empymod",
        "--level",
        "S0T0B0",
        "--output-root",
        str(tmp_path),
    ]

    assert cli.main(args) == 0
    assert cli.main(args) == 0

    manifests = sorted(tmp_path.glob("lei2023_noip/*/manifest.json"))
    assert len(manifests) == 2
    first = json.loads(manifests[0].read_text(encoding="utf-8"))
    second = json.loads(manifests[1].read_text(encoding="utf-8"))
    assert first["run_id"] != second["run_id"]
    assert first["schema"] == "atem3d.sotem.validation-manifest"
    assert first["schema_version"] == 1
    assert first["case_id"] == "lei2023_noip"
    assert first["case_path"] == str(LEI_CASE.resolve())
    assert len(first["case_file_sha256"]) == 64
    assert len(first["case_hash"]) == 64
    assert first["solver_id"] == "empymod"
    assert first["level"] == "S0T0B0"
    assert first["status"] == "prepared"
    assert first["git_commit"]
    assert type(first["git_dirty"]) is bool
    assert first["created_at"].endswith("Z")
    assert first["python_version"]
    assert {"numpy", "scipy", "simpeg", "empymod"} <= set(first["library_versions"])


def test_prepare_refuses_nonempty_explicit_directory_without_resume(tmp_path):
    run_dir = _prepare(tmp_path)
    before = (run_dir / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="non-empty|resume"):
        cli.main(
            [
                "prepare",
                "--case",
                str(LEI_CASE),
                "--solver",
                "empymod",
                "--run-dir",
                str(run_dir),
            ]
        )

    assert (run_dir / "manifest.json").read_bytes() == before
    assert cli.main(
        [
            "prepare",
            "--case",
            str(LEI_CASE),
            "--solver",
            "empymod",
            "--run-dir",
            str(run_dir),
            "--resume",
        ]
    ) == 0
    assert (run_dir / "manifest.json").read_bytes() == before


def test_nonprepare_requires_resume_for_existing_run(tmp_path):
    run_dir = _prepare(tmp_path)

    with pytest.raises(FileExistsError, match="resume"):
        cli.main(_command("reference", run_dir, LEI_CASE, resume=False))


def test_resume_rejects_changed_case_before_writing_or_calling_api(tmp_path, monkeypatch):
    case_path = tmp_path / "case.yaml"
    shutil.copyfile(LEI_CASE, case_path)
    run_dir = _prepare(tmp_path, case=case_path)
    manifest_before = (run_dir / "manifest.json").read_bytes()
    case_path.write_text(case_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    called = False

    def fake_reference(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("reference must not run")

    monkeypatch.setattr(cli, "get_empymod_reference", fake_reference)
    with pytest.raises(ValueError, match="hash"):
        cli.main(_command("reference", run_dir, case_path))

    assert called is False
    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "case_snapshot.yaml",
        "manifest.json",
        "model.json",
    ]


def test_resume_rejects_solver_mismatch_before_writing(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, solver="empymod")
    before = (run_dir / "manifest.json").read_bytes()
    monkeypatch.setattr(
        cli,
        "run_simpeg_benchmark",
        lambda *args, **kwargs: pytest.fail("simulation must not run"),
    )

    with pytest.raises(ValueError, match="solver"):
        cli.main(_command("simpeg", run_dir, LEI_CASE, extra=("--variant", "noip")))

    assert (run_dir / "manifest.json").read_bytes() == before


def test_reference_routes_exact_variant_and_writes_canonical_and_provenance(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, case=SONG_CASE, solver="empymod")
    captured = {}

    def fake_reference(times, config, *, mode):
        captured.update(times=np.asarray(times), config=config, mode=mode)
        case = cli.load_benchmark_case(SONG_CASE)
        return {**_fake_response(case), "reference_mode": mode}

    monkeypatch.setattr(cli, "get_empymod_reference", fake_reference)
    assert cli.main(
        _command(
            "reference",
            run_dir,
            SONG_CASE,
            extra=("--variant", "cole-cole-exact"),
        )
    ) == 0

    case = cli.load_benchmark_case(SONG_CASE)
    np.testing.assert_array_equal(captured["times"], case.observation_times)
    assert captured["mode"] == "cole-cole-exact"
    assert captured["config"].ramp_off_time == 0.0
    assert captured["config"].source_current == 10.0
    canonical = (run_dir / "empymod.csv").read_text(encoding="utf-8").splitlines()
    assert canonical[0] == (
        "time_obs_s,Ex_V_per_m,Ey_V_per_m,Hz_A_per_m,Bz_T,dBzdt_T_per_s"
    )
    assert (run_dir / "reference_empymod_or_1d.csv").is_file()
    metadata = json.loads((run_dir / "empymod_metadata.json").read_text(encoding="utf-8"))
    assert metadata["reference_mode"] == "cole-cole-exact"
    assert metadata["reference_provenance"] == "empymod_exact_layered"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "reference_complete"
    assert manifest["stages"]["reference"]["status"] == "complete"

    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda *args, **kwargs: pytest.fail("completed stage must be idempotent"),
    )
    assert cli.main(
        _command(
            "reference",
            run_dir,
            SONG_CASE,
            extra=("--variant", "cole-cole-exact"),
        )
    ) == 0


def test_simpeg_routes_stb_level_and_writes_honest_solver_metadata(tmp_path, monkeypatch):
    run_dir = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver=SIMPEG_SOLVER,
        level="S1T2B0",
    )
    captured = {}

    def fake_run(case, **kwargs):
        captured.update(case=case, kwargs=kwargs)
        return {
            **_fake_response(case, scale=2.0),
            "solver_id": SIMPEG_SOLVER,
            **_valid_simpeg_provenance(case, kwargs["variant"], "S1T2B0"),
            "variant": kwargs["variant"],
            "material_fit": _valid_ip_material_fit(),
        }

    monkeypatch.setattr(cli, "run_simpeg_benchmark", fake_run)
    assert cli.main(
        _command("simpeg", run_dir, SONG_CASE, extra=("--variant", "ip"))
    ) == 0

    assert captured["case"].case_id == "song2025_layered_pair"
    assert captured["kwargs"] == {
        "variant": "ip",
        "spatial_level": "S1",
        "boundary_level": "B0",
        "substeps": 4,
    }
    assert (run_dir / "simpeg.csv").is_file()
    assert (run_dir / "predictions.csv").is_file()
    metadata = json.loads((run_dir / "simpeg_metadata.json").read_text(encoding="utf-8"))
    assert metadata["solver_id"] == SIMPEG_SOLVER
    assert metadata["level"] == "S1T2B0"
    assert metadata["material_fit"]["material_gate_pass"] is True
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "simpeg_complete"


def test_effect_composes_four_completed_cli_runs_end_to_end(tmp_path, monkeypatch):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name="effect-run",
    )

    assert cli.main(_effect_args(effect_run, runs)) == 0

    output = effect_run / "effect"
    summary = json.loads(
        (output / "polarization_effect_summary.json").read_text(encoding="utf-8")
    )
    assert summary["definition"] == "ip_minus_noip"
    assert summary["threshold"] == 0.10
    assert (output / "polarization_effect_predictions.csv").is_file()
    assert (output / "polarization_effect_reference.csv").is_file()
    manifest = json.loads((effect_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "effect_complete"
    stage_inputs = manifest["stages"]["effect"]["inputs"]
    assert set(stage_inputs["source_runs"]) == {
        "noip_simpeg",
        "noip_reference",
        "ip_simpeg",
        "ip_reference",
    }
    assert all(
        len(item["evidence_file_sha256"]) == 64
        for item in stage_inputs["source_runs"].values()
    )


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing_directory",
        "same_directory",
        "nonpolarizable_case",
        "wrong_polarizable_case",
        "case_id",
        "case_hash",
        "level",
        "reference_solver",
        "simpeg_solver",
        "reference_variant",
        "simpeg_variant",
        "reference_stage",
        "simpeg_stage",
        "reference_file_hash",
        "simpeg_file_hash",
        "simpeg_case_snapshot_hash",
        "simpeg_model_hash",
        "simpeg_transaction_bundle",
        "simpeg_manifest_status",
    ],
)
def test_effect_rejects_mixed_or_tampered_source_runs_before_writing(
    tmp_path, monkeypatch, mismatch
):
    if mismatch == "wrong_polarizable_case":
        effect_case = tmp_path / "other-polarizable.yaml"
        effect_case.write_text(
            SONG_CASE.read_text(encoding="utf-8").replace(
                "case_id: song2025_layered_pair", "case_id: other_polarizable"
            ),
            encoding="utf-8",
        )
    else:
        effect_case = LEI_CASE if mismatch == "nonpolarizable_case" else SONG_CASE
    source_case = effect_case if mismatch == "wrong_polarizable_case" else SONG_CASE
    runs = _build_effect_source_runs(tmp_path, monkeypatch, case_path=source_case)
    effect_run = _prepare(
        tmp_path,
        case=effect_case,
        solver="polarization_effect",
        run_name="effect-run",
    )
    if mismatch == "missing_directory":
        runs["ip_reference"] = tmp_path / "missing-source-run"
    elif mismatch == "same_directory":
        runs["ip_reference"] = runs["noip_reference"]
    elif mismatch not in {
        "nonpolarizable_case",
        "wrong_polarizable_case",
        "reference_file_hash",
        "simpeg_file_hash",
        "simpeg_case_snapshot_hash",
        "simpeg_model_hash",
        "simpeg_transaction_bundle",
    }:
        role = "ip_simpeg" if mismatch.startswith("simpeg") else "ip_reference"
        manifest_path = runs[role] / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mismatch == "case_id":
            payload["case_id"] = "wrong_case"
        elif mismatch == "case_hash":
            payload["case_hash"] = "0" * 64
        elif mismatch == "level":
            payload["level"] = "S1T0B0"
        elif mismatch == "reference_solver":
            payload["solver_id"] = SIMPEG_SOLVER
        elif mismatch == "simpeg_solver":
            payload["solver_id"] = "empymod"
        elif mismatch == "reference_variant":
            payload["stages"]["reference"]["inputs"]["variant"] = "noip"
        elif mismatch == "simpeg_variant":
            payload["stages"]["simpeg"]["inputs"]["variant"] = "noip"
        elif mismatch == "reference_stage":
            del payload["stages"]["reference"]
        elif mismatch == "simpeg_stage":
            del payload["stages"]["simpeg"]
        elif mismatch == "simpeg_manifest_status":
            payload["status"] = "prepared"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    elif mismatch == "reference_file_hash":
        with (runs["ip_reference"] / "reference_empymod_or_1d.csv").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("tampered\n")
    elif mismatch == "simpeg_file_hash":
        with (runs["ip_simpeg"] / "predictions.csv").open("a", encoding="utf-8") as stream:
            stream.write("tampered\n")
    elif mismatch == "simpeg_case_snapshot_hash":
        with (runs["ip_simpeg"] / "case_snapshot.yaml").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write("# tampered\n")
    elif mismatch == "simpeg_model_hash":
        with (runs["ip_simpeg"] / "model.json").open("a", encoding="utf-8") as stream:
            stream.write("tampered\n")
    elif mismatch == "simpeg_transaction_bundle":
        journal = runs["ip_simpeg"] / "artifacts" / "simpeg" / "_transaction.json"
        journal.write_text("{}", encoding="utf-8")

    manifest_before = (effect_run / "manifest.json").read_bytes()
    called = False

    def forbidden_effect(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("effect API must not run for invalid source identities")

    monkeypatch.setattr(cli, "write_polarization_effect_artifacts", forbidden_effect)
    with pytest.raises((FileNotFoundError, ValueError)):
        cli.main(_effect_args(effect_run, runs, case=effect_case))

    assert called is False
    assert (effect_run / "manifest.json").read_bytes() == manifest_before
    assert not (effect_run / "effect").exists()
    assert not list(effect_run.glob(".effect-staging-*"))


def test_effect_threshold_is_fixed_and_cannot_be_changed_from_cli(tmp_path):
    run_dir = _prepare(tmp_path, case=SONG_CASE, solver="polarization_effect")

    with pytest.raises(SystemExit):
        cli.main(
            _command(
                "effect",
                run_dir,
                SONG_CASE,
                extra=(
                    "--noip-simpeg-run",
                    str(tmp_path / "noip-simpeg"),
                    "--noip-reference-run",
                    str(tmp_path / "noip-reference"),
                    "--ip-simpeg-run",
                    str(tmp_path / "ip-simpeg"),
                    "--ip-reference-run",
                    str(tmp_path / "ip-reference"),
                    "--threshold",
                    "0.2",
                ),
            )
        )


def test_finalize_missing_evidence_fails_closed_and_writes_json_safe_summary(tmp_path):
    run_dir = _prepare(tmp_path, solver="sotem_gate")

    assert cli.main(_command("finalize", run_dir, LEI_CASE)) == 0

    summary = json.loads((run_dir / "final_gate_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["failed_gates"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed_with_reproducible_evidence"


def test_finalize_routes_gate_mapping_and_preserves_nonfinite_evidence(tmp_path):
    run_dir = _prepare(tmp_path, solver="sotem_gate")
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps(
            {
                "lei_simpeg": True,
                "extra": {"measured_failure": "nan"},
                "reference_provenance": "empymod_exact_layered",
            }
        ),
        encoding="utf-8",
    )

    assert cli.main(
        _command("finalize", run_dir, LEI_CASE, extra=("--gates", str(gates)))
    ) == 0

    summary = json.loads((run_dir / "final_gate_summary.json").read_text(encoding="utf-8"))
    assert summary["gates"]["extra"] == {"measured_failure": "nan"}
    assert summary["state"] == "failed_with_reproducible_evidence"


def test_finalize_cannot_promote_hand_written_true_gates_to_validated(tmp_path):
    run_dir = _prepare(tmp_path, solver="sotem_gate", run_name="unverified-success")
    gates = tmp_path / "claimed-success.json"
    gate_names = (
        "lei_simpeg",
        "lei_fenicsx",
        "song_noip_simpeg",
        "song_noip_fenicsx",
        "song_ip_simpeg",
        "song_ip_fenicsx",
        "song_delta_simpeg",
        "song_delta_fenicsx",
        "material_gate",
    )
    gates.write_text(
        json.dumps(
            {
                **{name: True for name in gate_names},
                "reference_provenance": "empymod_exact_layered",
            }
        ),
        encoding="utf-8",
    )

    assert cli.main(
        _command("finalize", run_dir, LEI_CASE, extra=("--gates", str(gates)))
    ) == 0
    summary = json.loads((run_dir / "final_gate_summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["claimed_state"] == "ip_internally_validated"
    assert "unverified_external_gate_evidence" in summary["reason_codes"]


def test_finalize_snapshots_external_gates_for_resume(tmp_path):
    run_dir = _prepare(tmp_path, solver="sotem_gate", run_name="gate-snapshot")
    gates = tmp_path / "gates-to-remove.json"
    original = json.dumps({"lei_simpeg": False}).encode("utf-8")
    gates.write_bytes(original)
    args = _command("finalize", run_dir, LEI_CASE, extra=("--gates", str(gates)))
    assert cli.main(args) == 0
    snapshot = run_dir / "artifacts" / "finalize" / "inputs" / "gates.json"
    assert snapshot.read_bytes() == original
    summary_before = (run_dir / "final_gate_summary.json").read_bytes()
    gates.unlink()

    assert cli.main(args) == 0
    assert (run_dir / "final_gate_summary.json").read_bytes() == summary_before


def test_resume_rejects_modified_completed_stage_without_overwrite(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path)
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )
    args = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))
    assert cli.main(args) == 0
    output = run_dir / "empymod.csv"
    output.write_text("tampered evidence\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash|evidence"):
        cli.main(args)

    assert output.read_text(encoding="utf-8") == "tampered evidence\n"


def test_simpeg_nonfinite_response_is_rejected_before_output(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, solver=SIMPEG_SOLVER)
    case = cli.load_benchmark_case(LEI_CASE)
    result = {
        **_fake_response(case),
        "solver_id": SIMPEG_SOLVER,
        "mesh_hash": "a" * 64,
        "time_hash": "b" * 64,
        "mesh_stats": {},
        "variant": "noip",
        "material_fit": None,
    }
    result["data"][0, 0] = np.nan
    monkeypatch.setattr(cli, "run_simpeg_benchmark", lambda *args, **kwargs: result)

    with pytest.raises(ValueError, match="finite"):
        cli.main(_command("simpeg", run_dir, LEI_CASE, extra=("--variant", "noip")))

    assert not (run_dir / "simpeg.csv").exists()
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "prepared"


@pytest.mark.parametrize(
    ("defect", "value"),
    [
        ("mesh_hash", None),
        ("time_hash", "not-a-hash"),
        ("n_cells", True),
        ("n_edges", 0),
        ("mesh_stats_hash", "c" * 64),
        ("coordinate_system", None),
        ("material_fit", None),
        ("material_gate_pass", 1),
        ("fit_term_count", 15),
        ("relative_l2", 0.02),
        ("dc_residual", float("inf")),
        ("minimum_delta_sigma", 0.0),
        ("debye_terms", []),
    ],
)
def test_simpeg_provenance_defects_fail_before_stage_output(
    tmp_path, monkeypatch, defect, value
):
    run_dir = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver=SIMPEG_SOLVER,
        run_name="simpeg-invalid",
    )
    case = cli.load_benchmark_case(SONG_CASE)
    result = {
        **_fake_response(case),
        "solver_id": SIMPEG_SOLVER,
        **_valid_simpeg_provenance(case, "ip"),
        "variant": "ip",
        "material_fit": _valid_ip_material_fit(),
    }
    if defect in {"mesh_hash", "time_hash", "coordinate_system", "material_fit"}:
        result[defect] = value
    elif defect == "mesh_stats_hash":
        result["mesh_stats"]["mesh_hash"] = value
    elif defect in {"n_cells", "n_edges"}:
        result["mesh_stats"][defect] = value
    else:
        result["material_fit"][defect] = value
    monkeypatch.setattr(cli, "run_simpeg_benchmark", lambda *args, **kwargs: result)
    manifest_before = (run_dir / "manifest.json").read_bytes()

    with pytest.raises(ValueError):
        cli.main(_command("simpeg", run_dir, SONG_CASE, extra=("--variant", "ip")))

    assert (run_dir / "manifest.json").read_bytes() == manifest_before
    assert not (run_dir / "artifacts" / "simpeg").exists()
    assert not (run_dir / "simpeg.csv").exists()


def test_simpeg_hashes_are_bound_to_the_requested_mesh_and_time_schedule(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, solver=SIMPEG_SOLVER, run_name="unbound-hash")
    case = cli.load_benchmark_case(LEI_CASE)
    result = {
        **_fake_response(case),
        "solver_id": SIMPEG_SOLVER,
        **_valid_simpeg_provenance(case, "noip"),
        "variant": "noip",
        "material_fit": None,
    }
    result["mesh_hash"] = "c" * 64
    result["mesh_stats"]["mesh_hash"] = "c" * 64
    monkeypatch.setattr(cli, "run_simpeg_benchmark", lambda *args, **kwargs: result)

    with pytest.raises(ValueError, match="mesh|hash|provenance"):
        cli.main(_command("simpeg", run_dir, LEI_CASE, extra=("--variant", "noip")))


@pytest.mark.parametrize(
    "defect", ["coordinate_transform", "conductivity_balance", "requested_material"]
)
def test_simpeg_rejects_incomplete_or_inconsistent_physical_provenance(
    tmp_path, monkeypatch, defect
):
    run_dir = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver=SIMPEG_SOLVER,
        run_name=f"physical-{defect}",
    )
    case = cli.load_benchmark_case(SONG_CASE)
    result = {
        **_fake_response(case),
        "solver_id": SIMPEG_SOLVER,
        **_valid_simpeg_provenance(case, "ip"),
        "variant": "ip",
        "material_fit": _valid_ip_material_fit(),
    }
    if defect == "coordinate_transform":
        result.pop("coordinate_transform")
    elif defect == "conductivity_balance":
        result["material_fit"]["delta_sum"] *= 2.0
    else:
        for term in result["material_fit"]["debye_terms"]:
            term["tau"] *= 2.0
    monkeypatch.setattr(cli, "run_simpeg_benchmark", lambda *args, **kwargs: result)

    with pytest.raises(ValueError, match="coordinate|conduct|delta|provenance"):
        cli.main(_command("simpeg", run_dir, SONG_CASE, extra=("--variant", "ip")))


@pytest.mark.parametrize(
    "unsafe_name",
    ["../outside.txt", "/absolute/outside.txt", r"C:\\outside.txt", "./inside.txt"],
)
def test_manifest_evidence_paths_must_be_safe_relative_paths(tmp_path, unsafe_name):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("evidence", encoding="utf-8")
    manifest = {
        "stages": {
            "reference": {
                "status": "complete",
                "inputs": {"variant": "noip"},
                "file_sha256": {unsafe_name: cli._sha256_file(outside)},
            }
        }
    }

    with pytest.raises(ValueError, match="path|evidence|relative"):
        cli._completed_stage_is_intact(
            run_dir, manifest, "reference", {"variant": "noip"}
        )


def test_manifest_evidence_rejects_symlink_to_outside(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("evidence", encoding="utf-8")
    link = run_dir / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    manifest = {
        "stages": {
            "reference": {
                "status": "complete",
                "inputs": {"variant": "noip"},
                "file_sha256": {"linked.txt": cli._sha256_file(outside)},
            }
        }
    }

    with pytest.raises(ValueError, match="symlink|outside|path"):
        cli._completed_stage_is_intact(
            run_dir, manifest, "reference", {"variant": "noip"}
        )


def test_stage_publication_rejects_symlinked_artifacts_before_writing(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="symlinked-artifacts")
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    link = run_dir / "artifacts"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsafe publication reached the solver")

    monkeypatch.setattr(cli, "get_empymod_reference", forbidden)
    with pytest.raises(ValueError, match="symlink|junction|outside|artifacts|path"):
        cli.main(_command("reference", run_dir, LEI_CASE))

    assert called is False
    assert not list(outside.iterdir())


def test_effect_detects_source_replacement_between_validation_and_copy(
    tmp_path, monkeypatch
):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name="effect-run",
    )
    source = runs["noip_simpeg"] / "predictions.csv"
    original_copy = shutil.copyfile
    replaced = False

    def replace_then_copy(src, dst, *args, **kwargs):
        nonlocal replaced
        if Path(src) == source and not replaced:
            source.write_text("replaced after validation\n", encoding="utf-8")
            replaced = True
        return original_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(cli.shutil, "copyfile", replace_then_copy)
    monkeypatch.setattr(
        cli,
        "write_polarization_effect_artifacts",
        lambda *args, **kwargs: pytest.fail("TOCTOU replacement must fail before effect API"),
    )
    manifest_before = (effect_run / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="hash|changed|source"):
        cli.main(_effect_args(effect_run, runs))

    assert replaced is True
    assert (effect_run / "manifest.json").read_bytes() == manifest_before
    assert not (effect_run / "artifacts" / "effect").exists()


def test_gates_are_hashed_and_parsed_from_one_read(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, solver="sotem_gate", run_name="gate-run")
    gates = tmp_path / "gates.json"
    original = json.dumps({"reference_provenance": "empymod_exact_layered"}).encode()
    gates.write_bytes(original)
    original_read = Path.read_bytes
    reads = 0

    def counted_read(path):
        nonlocal reads
        if path == gates:
            reads += 1
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    assert cli.main(
        _command("finalize", run_dir, LEI_CASE, extra=("--gates", str(gates)))
    ) == 0
    assert reads == 1


def test_stage_bundle_recovers_manifest_commit_after_interruption(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="recover-run")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )
    original_record = cli._record_stage
    monkeypatch.setattr(
        cli,
        "_record_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    args = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))

    with pytest.raises(RuntimeError, match="interrupted"):
        cli.main(args)

    assert (run_dir / "artifacts" / "reference" / "_transaction.json").is_file()
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "prepared"
    monkeypatch.setattr(cli, "_record_stage", original_record)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda *args, **kwargs: pytest.fail("resume must recover the published bundle"),
    )

    assert cli.main(args) == 0
    assert json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"] == "reference_complete"


def test_completed_stage_restores_a_missing_export_from_its_bundle(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="missing-export-run")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )
    args = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))
    assert cli.main(args) == 0
    output = run_dir / "empymod.csv"
    expected = output.read_bytes()
    output.unlink()
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda *args, **kwargs: pytest.fail("resume must use the transaction bundle"),
    )

    assert cli.main(args) == 0
    assert output.read_bytes() == expected


def test_completed_stage_rejects_corrupted_transaction_bundle(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="corrupt-bundle-run")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )
    args = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))
    assert cli.main(args) == 0
    (run_dir / "artifacts" / "reference" / "empymod.csv").write_text(
        "corrupted transaction evidence\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda *args, **kwargs: pytest.fail("resume must validate the published bundle"),
    )

    with pytest.raises(ValueError, match="bundle|hash|evidence"):
        cli.main(args)


def test_cross_process_run_lock_rejects_competing_writer(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="locked-run")
    marker = tmp_path / "locked.marker"
    release = tmp_path / "release.marker"
    script = (
        "import sys,time; from pathlib import Path; "
        "from atem3d.sotem_validation_cli import _run_lock; "
        "run=Path(sys.argv[1]); marker=Path(sys.argv[2]); release=Path(sys.argv[3]); "
        "\nwith _run_lock(run):\n marker.write_text('locked');\n"
        " while not release.exists(): time.sleep(0.01)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(run_dir), str(marker), str(release)],
        env=env,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.is_file()
        called = False

        def forbidden(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("competing writer reached stage API")

        monkeypatch.setattr(cli, "get_empymod_reference", forbidden)
        with pytest.raises(RuntimeError, match="lock|writer|active"):
            cli.main(_command("reference", run_dir, LEI_CASE))
        assert called is False
    finally:
        release.write_text("release", encoding="utf-8")
        process.wait(timeout=10)
    with cli._run_lock(run_dir):
        pass


def test_manifest_and_stage_record_environment_and_resource_fields(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="resource-run")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    environment = manifest["environment"]
    assert all(
        type(environment[name]) is str
        for name in (
            "platform",
            "os_name",
            "machine",
            "sys_executable",
            "python_implementation",
            "cwd",
            "conda_env",
            "conda_prefix",
            "wsl_distribution",
        )
    )
    assert type(environment["is_wsl"]) is bool
    prepare_record = manifest["prepare"]
    assert prepare_record["elapsed_seconds"] >= 0.0
    assert type(prepare_record["process_peak_rss_bytes"]) is int
    assert prepare_record["process_peak_rss_bytes"] > 0
    assert prepare_record["peak_rss_scope"] == "process_global_peak"

    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )
    assert cli.main(_command("reference", run_dir, LEI_CASE)) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["reference"]
    assert stage["started_at"].endswith("Z")
    assert stage["completed_at"].endswith("Z")
    assert stage["elapsed_seconds"] >= 0.0
    assert type(stage["process_peak_rss_bytes"]) is int
    assert stage["process_peak_rss_bytes"] > 0
    assert stage["peak_rss_scope"] == "process_global_peak"


def test_stage_records_its_own_execution_environment(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="stage-environment")
    stage_environment = {
        "platform": "stage-platform",
        "os_name": "stage-os",
        "machine": "stage-machine",
        "sys_executable": "/stage/python",
        "python_implementation": "CPython",
        "cwd": "/stage/cwd",
        "conda_env": "fenicsx",
        "conda_prefix": "/stage/conda",
        "is_wsl": True,
        "wsl_distribution": "Ubuntu",
    }
    monkeypatch.setattr(cli, "_environment_metadata", lambda: stage_environment)
    monkeypatch.setattr(cli, "_library_versions", lambda: {"numpy": "stage-version"})
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )

    assert cli.main(_command("reference", run_dir, LEI_CASE)) == 0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stage = manifest["stages"]["reference"]
    assert stage["environment"] == stage_environment
    assert stage["python_version"] == sys.version.split()[0]
    assert stage["library_versions"] == {"numpy": "stage-version"}


def test_stage_publication_flushes_parent_directories(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="durable-stage")
    flushed = []
    monkeypatch.setattr(
        cli,
        "_fsync_directory",
        lambda path: flushed.append(Path(path).resolve()),
        raising=False,
    )
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )

    assert cli.main(_command("reference", run_dir, LEI_CASE)) == 0
    assert run_dir.resolve() in flushed
    assert (run_dir / "artifacts").resolve() in flushed


@pytest.mark.parametrize("level", ["S0B0", "S3T0B0", "S0T9B0", "S0T0B3", "s0t0b0"])
def test_prepare_rejects_invalid_level_without_creating_output(tmp_path, level):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "prepare",
                "--case",
                str(LEI_CASE),
                "--solver",
                "empymod",
                "--level",
                level,
                "--output-root",
                str(tmp_path),
            ]
        )
    assert not list(tmp_path.rglob("manifest.json"))


def test_prepare_rejects_empty_solver_id_without_creating_output(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(
            [
                "prepare",
                "--case",
                str(LEI_CASE),
                "--solver",
                "",
                "--output-root",
                str(tmp_path),
            ]
        )
    assert not list(tmp_path.rglob("manifest.json"))


def test_case_identity_hashes_and_parses_one_immutable_snapshot(tmp_path, monkeypatch):
    case_path = tmp_path / "case.yaml"
    original = LEI_CASE.read_bytes()
    replacement = original.replace(b"case_id: lei2023_noip", b"case_id: replaced_case")
    case_path.write_bytes(original)
    original_read = Path.read_bytes
    reads = 0

    def replace_after_read(path):
        nonlocal reads
        payload = original_read(path)
        if path == case_path:
            reads += 1
            path.write_bytes(replacement)
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    normalized, case, identity, source_bytes = cli._case_identity(case_path)

    assert normalized == case_path.resolve()
    assert reads == 1
    assert case.case_id == "lei2023_noip"
    assert identity["case_file_sha256"] == cli._sha256_bytes(original)
    assert source_bytes == original


def test_prepare_preserves_model_and_resumes_after_case_source_is_removed(
    tmp_path, monkeypatch
):
    case_path = tmp_path / "source-case.yaml"
    original = LEI_CASE.read_bytes()
    case_path.write_bytes(original)
    run_dir = _prepare(tmp_path, case=case_path, run_name="preserved-case")
    assert (run_dir / "case_snapshot.yaml").read_bytes() == original
    model = json.loads((run_dir / "model.json").read_text(encoding="utf-8"))
    assert model["case_id"] == "lei2023_noip"
    case_path.unlink()
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode: {**_fake_response(case), "reference_mode": mode},
    )

    assert cli.main(_command("reference", run_dir, case_path)) == 0


def test_prepare_resume_uses_preserved_case_when_source_is_removed(tmp_path):
    case_path = tmp_path / "source-case.yaml"
    case_path.write_bytes(LEI_CASE.read_bytes())
    run_dir = _prepare(tmp_path, case=case_path, run_name="prepare-resume")
    manifest_before = (run_dir / "manifest.json").read_bytes()
    case_path.unlink()

    assert cli.main(
        [
            "prepare",
            "--case",
            str(case_path),
            "--solver",
            "empymod",
            "--level",
            "S0T0B0",
            "--run-dir",
            str(run_dir),
            "--resume",
        ]
    ) == 0
    assert (run_dir / "manifest.json").read_bytes() == manifest_before


def test_reference_rejects_invalid_variant_and_missing_paths(tmp_path):
    run_dir = _prepare(tmp_path)
    with pytest.raises(SystemExit):
        cli.main(_command("reference", run_dir, LEI_CASE, extra=("--variant", "ip")))
    with pytest.raises(FileNotFoundError):
        cli.main(
            _command(
                "reference",
                tmp_path / "missing-run",
                LEI_CASE,
                extra=("--variant", "noip"),
            )
        )
    with pytest.raises(FileNotFoundError):
        cli.main(
            _command(
                "reference",
                run_dir,
                tmp_path / "missing-case.yaml",
                extra=("--variant", "noip"),
            )
        )


def test_pyproject_registers_sotem_validation_entrypoint():
    payload = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'atem3d-sotem-validate = "atem3d.sotem_validation_cli:main"' in payload
