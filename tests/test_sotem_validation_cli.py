from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import atem3d.sotem_validation_cli as cli


ROOT = Path(__file__).resolve().parents[1]
LEI_CASE = ROOT / "benchmarks" / "sotem" / "lei2023_noip.yaml"
SONG_CASE = ROOT / "benchmarks" / "sotem" / "song2025_layered_pair.yaml"
SIMPEG_SOLVER = "atem3d_simpeg_discretize_debye"


def _prepare(tmp_path, case=LEI_CASE, *, solver="empymod", level="S0T0B0"):
    run_dir = tmp_path / "run"
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
    assert sorted(path.name for path in run_dir.iterdir()) == ["manifest.json"]


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
            "mesh_hash": "a" * 64,
            "time_hash": "b" * 64,
            "mesh_stats": {"n_cells": 8, "n_edges": 54},
            "variant": kwargs["variant"],
            "material_fit": {"material_gate_pass": True},
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


def test_effect_routes_complete_pair_without_transforming_paths(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, case=SONG_CASE, solver="polarization_effect")
    noip_dir = tmp_path / "noip"
    ip_dir = tmp_path / "ip"
    noip_dir.mkdir()
    ip_dir.mkdir()
    for directory in (noip_dir, ip_dir):
        (directory / "predictions.csv").write_text("complete", encoding="utf-8")
        (directory / "reference_empymod_or_1d.csv").write_text("complete", encoding="utf-8")
    captured = {}

    def fake_effect(noip, ip, output, *, threshold):
        captured.update(noip=Path(noip), ip=Path(ip), output=Path(output), threshold=threshold)
        Path(output).mkdir(parents=True)
        (Path(output) / "polarization_effect_summary.json").write_text(
            json.dumps({"passed": False, "definition": "ip_minus_noip"}),
            encoding="utf-8",
        )
        return {"passed": False, "definition": "ip_minus_noip"}

    monkeypatch.setattr(cli, "write_polarization_effect_artifacts", fake_effect)
    assert cli.main(
        _command(
            "effect",
            run_dir,
            SONG_CASE,
            extra=("--noip-dir", str(noip_dir), "--ip-dir", str(ip_dir)),
        )
    ) == 0

    assert captured == {
        "noip": noip_dir.resolve(),
        "ip": ip_dir.resolve(),
        "output": (run_dir / "effect").resolve(),
        "threshold": 0.10,
    }
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "effect_complete"


def test_effect_requires_complete_noip_ip_pair_before_api_call(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, case=SONG_CASE, solver="polarization_effect")
    noip_dir = tmp_path / "noip"
    ip_dir = tmp_path / "ip"
    noip_dir.mkdir()
    ip_dir.mkdir()
    monkeypatch.setattr(
        cli,
        "write_polarization_effect_artifacts",
        lambda *args, **kwargs: pytest.fail("incomplete pair must not run"),
    )

    with pytest.raises(FileNotFoundError, match="predictions|reference"):
        cli.main(
            _command(
                "effect",
                run_dir,
                SONG_CASE,
                extra=("--noip-dir", str(noip_dir), "--ip-dir", str(ip_dir)),
            )
        )


def test_effect_threshold_is_fixed_and_cannot_be_changed_from_cli(tmp_path):
    run_dir = _prepare(tmp_path, case=SONG_CASE, solver="polarization_effect")

    with pytest.raises(SystemExit):
        cli.main(
            _command(
                "effect",
                run_dir,
                SONG_CASE,
                extra=(
                    "--noip-dir",
                    str(tmp_path / "noip"),
                    "--ip-dir",
                    str(tmp_path / "ip"),
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
