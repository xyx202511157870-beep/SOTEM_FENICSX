from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

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


def _valid_initialization_diagnostics():
    diagnostics = [
        {
            "phase": phase,
            "solver": solver,
            "solve_mode": "petsc_ksp",
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "backend_reason": 2,
            "backend_reported_converged": True,
            "backend_iterations": 3,
            "external_true_relative_residual": 1.0e-12,
            "external_tolerance": 1.0e-8,
            "internal_tolerance": 1.0e-11,
            "residual_replacement_steps": 0,
            "balance_name": balance_name,
            "balance_relative_residual": 1.0e-12,
            "balance_tolerance": 1.0e-8,
        }
        for phase, solver, ksp_type, pc_type, balance_name in (
            (
                "dc_electric",
                "petsc_ksp_hypre_boomeramg",
                "cg",
                "hypre_boomeramg",
                "discrete_current_divergence",
            ),
            (
                "ampere_magnetic",
                "petsc_ksp_hypre_ams",
                "gmres",
                "hypre_ams",
                "static_ampere",
            ),
        )
    ]
    diagnostics[1].update(
        gauge_stabilization_weight=5.0e14,
        stiffness_operator_max_abs=2.0e12,
        gauge_operator_max_abs=4.0e-3,
    )
    return diagnostics


_GAUGE_EVIDENCE_FIELDS = (
    "gauge_stabilization_weight",
    "stiffness_operator_max_abs",
    "gauge_operator_max_abs",
)


def _set_exact_zero_initialization_diagnostics(diagnostics):
    for record in diagnostics:
        record.update(
            solve_mode="exact_zero_rhs",
            backend_reason=0,
            backend_reported_converged=False,
            backend_iterations=0,
            external_true_relative_residual=0.0,
            residual_replacement_steps=0,
            balance_relative_residual=0.0,
        )
        for field in _GAUGE_EVIDENCE_FIELDS:
            record.pop(field, None)


def _valid_linear_diagnostics(config):
    return [
        {
            "step_index": index,
            "dt_s": float(dt),
            "solver": "petsc_ksp_hypre_ams",
            "solve_mode": "petsc_ksp",
            "ksp_type": config["solver"]["ksp_type"],
            "pc_type": "hypre_ams",
            "backend_reason": 2,
            "backend_reported_converged": True,
            "backend_iterations": 3,
            "external_true_relative_residual": 1.0e-12,
            "external_tolerance": 1.0e-8,
            "internal_tolerance": 1.0e-11,
            "residual_replacement_steps": 0,
        }
        for index, dt in enumerate(config["time_steps"])
    ]


def _lei_noip_config():
    case = cli.load_benchmark_case(LEI_CASE)
    return cli.build_benchmark_config(
        case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )


def test_publication_validator_rejects_negative_linear_residual():
    config = _lei_noip_config()
    diagnostics = _valid_linear_diagnostics(config)
    diagnostics[0]["external_true_relative_residual"] = -1.0e-12

    with pytest.raises(ValueError, match="linear solver diagnostics"):
        cli._validated_linear_diagnostics_for_publication(diagnostics, config)


def test_publication_validator_accepts_exact_zero_linear_diagnostics():
    config = _lei_noip_config()
    diagnostics = _valid_linear_diagnostics(config)
    for record in diagnostics:
        record.update(
            solve_mode="exact_zero_rhs",
            backend_reason=0,
            backend_reported_converged=False,
            backend_iterations=0,
            external_true_relative_residual=0.0,
            residual_replacement_steps=0,
        )

    validated = cli._validated_linear_diagnostics_for_publication(
        diagnostics,
        config,
    )

    assert len(validated) == len(config["time_steps"])
    assert all(item["solve_mode"] == "exact_zero_rhs" for item in validated)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_reason", 1),
        ("backend_reported_converged", True),
        ("backend_iterations", 1),
        ("external_true_relative_residual", 1.0e-12),
        ("residual_replacement_steps", 1),
    ],
)
def test_publication_validator_rejects_incoherent_exact_zero_linear_diagnostics(
    field,
    value,
):
    config = _lei_noip_config()
    diagnostics = _valid_linear_diagnostics(config)
    for record in diagnostics:
        record.update(
            solve_mode="exact_zero_rhs",
            backend_reason=0,
            backend_reported_converged=False,
            backend_iterations=0,
            external_true_relative_residual=0.0,
            residual_replacement_steps=0,
        )
    diagnostics[0][field] = value

    with pytest.raises(ValueError, match="linear solver diagnostics"):
        cli._validated_linear_diagnostics_for_publication(diagnostics, config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_reason", 1),
        ("backend_reported_converged", True),
        ("backend_iterations", 1),
        ("external_true_relative_residual", 1.0e-12),
        ("residual_replacement_steps", 1),
        ("balance_relative_residual", 1.0e-12),
    ],
)
def test_publication_validator_rejects_incoherent_exact_zero_diagnostics(
    field,
    value,
):
    case = cli.load_benchmark_case(LEI_CASE)
    config = cli.build_benchmark_config(
        case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    diagnostics = _valid_initialization_diagnostics()
    _set_exact_zero_initialization_diagnostics(diagnostics)
    diagnostics[0][field] = value

    with pytest.raises(ValueError, match="initialization diagnostics"):
        cli._validated_initialization_diagnostics_for_publication(
            diagnostics,
            config,
        )


@pytest.mark.parametrize(
    "injected",
    [
        {"gauge_stabilization_weight": 5.0e14},
        {"stiffness_operator_max_abs": 2.0e12},
        {"gauge_operator_max_abs": 4.0e-3},
        {
            "gauge_stabilization_weight": 5.0e14,
            "stiffness_operator_max_abs": 2.0e12,
            "gauge_operator_max_abs": 4.0e-3,
        },
    ],
)
def test_publication_rejects_gauge_evidence_on_exact_zero_initialization(
    injected,
):
    config = _lei_noip_config()
    diagnostics = _valid_initialization_diagnostics()
    _set_exact_zero_initialization_diagnostics(diagnostics)
    diagnostics[1].update(injected)

    with pytest.raises(ValueError, match="initialization diagnostics"):
        cli._validated_initialization_diagnostics_for_publication(
            diagnostics,
            config,
        )


@pytest.mark.parametrize("solve_mode", ["petsc_ksp", "exact_zero_rhs"])
@pytest.mark.parametrize(
    "injected",
    [
        {"gauge_stabilization_weight": 5.0e14},
        {"stiffness_operator_max_abs": 2.0e12},
        {"gauge_operator_max_abs": 4.0e-3},
        {
            "gauge_stabilization_weight": 5.0e14,
            "stiffness_operator_max_abs": 2.0e12,
            "gauge_operator_max_abs": 4.0e-3,
        },
    ],
)
def test_publication_rejects_gauge_evidence_on_dc_initialization(
    solve_mode,
    injected,
):
    config = _lei_noip_config()
    diagnostics = _valid_initialization_diagnostics()
    if solve_mode == "exact_zero_rhs":
        _set_exact_zero_initialization_diagnostics(diagnostics)
    diagnostics[0].update(injected)

    with pytest.raises(ValueError, match="initialization diagnostics"):
        cli._validated_initialization_diagnostics_for_publication(
            diagnostics,
            config,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gauge_stabilization_weight", None),
        ("gauge_stabilization_weight", np.inf),
        ("stiffness_operator_max_abs", -1.0),
        ("gauge_operator_max_abs", 0.0),
        ("gauge_stabilization_weight", 4.0e14),
    ],
)
def test_publication_validator_rejects_invalid_gauge_stabilization_evidence(
    field,
    value,
):
    case = cli.load_benchmark_case(LEI_CASE)
    config = cli.build_benchmark_config(
        case,
        variant="noip",
        spatial_level="S0",
        boundary_level="B0",
        substeps=1,
    )
    diagnostics = _valid_initialization_diagnostics()
    diagnostics[1][field] = value

    with pytest.raises(ValueError, match="initialization diagnostics"):
        cli._validated_initialization_diagnostics_for_publication(
            diagnostics,
            config,
        )


def _build_effect_source_runs(
    tmp_path, monkeypatch, *, level="S0T0B0", case_path=SONG_CASE
):
    case = cli.load_benchmark_case(case_path)

    def fake_reference(times, config, *, mode, srcpts):
        scale = 1.0 if mode == "noip" else 1.5
        return {**_fake_response(case, scale=scale), "reference_mode": mode}

    def fake_simpeg(case_value, **kwargs):
        scale = 1.1 if kwargs["variant"] == "noip" else 1.6
        config = cli.build_benchmark_config(case_value, **kwargs)
        return {
            **_fake_response(case_value, scale=scale),
            "solver_id": SIMPEG_SOLVER,
            **_valid_simpeg_provenance(case_value, kwargs["variant"], level),
            "variant": kwargs["variant"],
            "material_fit": (
                _valid_ip_material_fit() if kwargs["variant"] == "ip" else None
            ),
            "initialization_solver_diagnostics": _valid_initialization_diagnostics(),
            "linear_solver_diagnostics": _valid_linear_diagnostics(config),
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

    def fake_reference(times, config, *, mode, srcpts):
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
        config = cli.build_benchmark_config(case, **kwargs)
        return {
            **_fake_response(case, scale=2.0),
            "solver_id": SIMPEG_SOLVER,
            **_valid_simpeg_provenance(case, kwargs["variant"], "S1T2B0"),
            "variant": kwargs["variant"],
            "material_fit": _valid_ip_material_fit(),
            "initialization_solver_diagnostics": _valid_initialization_diagnostics(),
            "linear_solver_diagnostics": _valid_linear_diagnostics(config),
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
    assert metadata["solver_configuration"]["initialization"]["type"] == "petsc_hypre"
    assert metadata["solver_configuration"]["transient"]["type"] == "petsc_ams"
    assert [
        item["phase"] for item in metadata["initialization_solver_diagnostics"]
    ] == ["dc_electric", "ampere_magnetic"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "simpeg_complete"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("solve_mode", "unknown"),
        ("ksp_type", "wrong"),
        ("pc_type", "wrong"),
        ("balance_name", "wrong"),
        ("backend_reason", 0),
        ("external_true_relative_residual", -1.0e-12),
        ("external_true_relative_residual", 1.01e-8),
        ("balance_relative_residual", -1.0e-12),
        ("internal_tolerance", 2.0e-11),
        ("residual_replacement_steps", 3),
    ],
)
def test_simpeg_publication_revalidates_initialization_diagnostics(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    run_dir = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver=SIMPEG_SOLVER,
        level="S1T2B0",
    )

    def fake_run(case, **kwargs):
        diagnostics = _valid_initialization_diagnostics()
        diagnostics[0][field] = value
        config = cli.build_benchmark_config(case, **kwargs)
        return {
            **_fake_response(case, scale=2.0),
            "solver_id": SIMPEG_SOLVER,
            **_valid_simpeg_provenance(case, kwargs["variant"], "S1T2B0"),
            "variant": kwargs["variant"],
            "material_fit": _valid_ip_material_fit(),
            "initialization_solver_diagnostics": diagnostics,
            "linear_solver_diagnostics": _valid_linear_diagnostics(config),
        }

    monkeypatch.setattr(cli, "run_simpeg_benchmark", fake_run)

    with pytest.raises(ValueError, match="initialization diagnostics"):
        cli.main(_command("simpeg", run_dir, SONG_CASE, extra=("--variant", "ip")))

    assert not (run_dir / "simpeg.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("external_true_relative_residual", -1.0e-12),
        ("backend_reason", 0),
        ("backend_reported_converged", False),
    ],
)
def test_simpeg_publication_revalidates_linear_solver_diagnostics(
    tmp_path,
    monkeypatch,
    field,
    value,
):
    run_dir = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver=SIMPEG_SOLVER,
        level="S1T2B0",
    )

    def fake_run(case, **kwargs):
        config = cli.build_benchmark_config(case, **kwargs)
        diagnostics = _valid_linear_diagnostics(config)
        diagnostics[0][field] = value
        return {
            **_fake_response(case, scale=2.0),
            "solver_id": SIMPEG_SOLVER,
            **_valid_simpeg_provenance(case, kwargs["variant"], "S1T2B0"),
            "variant": kwargs["variant"],
            "material_fit": _valid_ip_material_fit(),
            "initialization_solver_diagnostics": _valid_initialization_diagnostics(),
            "linear_solver_diagnostics": diagnostics,
        }

    monkeypatch.setattr(cli, "run_simpeg_benchmark", fake_run)

    with pytest.raises(ValueError, match="linear solver diagnostics"):
        cli.main(_command("simpeg", run_dir, SONG_CASE, extra=("--variant", "ip")))

    assert not (run_dir / "simpeg.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"


def test_effect_composes_four_completed_cli_runs_end_to_end(tmp_path, monkeypatch):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name="effect-run",
    )

    assert cli.main(_effect_args(effect_run, runs)) == 0
    monkeypatch.setattr(
        cli,
        "write_polarization_effect_artifacts",
        lambda *args, **kwargs: pytest.fail(
            "completed effect recovery must not rerun the effect writer"
        ),
    )
    assert cli.main(_effect_args(effect_run, runs)) == 0

    output = effect_run / "effect"
    summary = json.loads(
        (output / "polarization_effect_summary.json").read_text(encoding="utf-8")
    )
    assert summary["definition"] == "ip_minus_noip"
    assert summary["threshold"] == 0.10
    assert summary["reference_srcpts"] == 17
    assert (output / "polarization_effect_predictions.csv").is_file()
    assert (output / "polarization_effect_reference.csv").is_file()
    manifest = json.loads((effect_run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "effect_complete"
    stage_inputs = manifest["stages"]["effect"]["inputs"]
    assert stage_inputs["reference_srcpts"] == 17
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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


def test_orphan_reference_bundle_is_semantically_validated_before_recovery(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="invalid-orphan-reference")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    original_record = cli._record_stage
    monkeypatch.setattr(
        cli,
        "_record_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    command = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))
    with pytest.raises(RuntimeError, match="interrupted"):
        cli.main(command)

    root_exports = [
        run_dir / "empymod.csv",
        run_dir / "reference_empymod_or_1d.csv",
        run_dir / "empymod_metadata.json",
    ]
    for output in root_exports:
        output.unlink()
    bundle_metadata = run_dir / "artifacts" / "reference" / "empymod_metadata.json"
    metadata = json.loads(bundle_metadata.read_text(encoding="utf-8"))
    metadata["srcpts"] = 9
    bundle_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["file_sha256"]["artifacts/reference/empymod_metadata.json"] = (
        cli._sha256_file(bundle_metadata)
    )
    transaction_path.write_text(
        json.dumps(transaction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = run_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    monkeypatch.setattr(cli, "_record_stage", original_record)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda *args, **kwargs: pytest.fail("invalid orphan must not rerun solver"),
    )

    with pytest.raises(ValueError, match="metadata|srcpts|identity"):
        cli.main(command)

    assert all(not output.exists() for output in root_exports)
    assert manifest_path.read_bytes() == manifest_before


def test_bad_reference_does_not_restore_missing_root_export(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="bad-ref")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))
    assert cli.main(command) == 0
    missing_output = run_dir / "empymod.csv"
    missing_output.unlink()
    _rewrite_reference_metadata_srcpts(run_dir, 9)
    manifest_path = run_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="metadata|srcpts|identity"):
        cli.main(command)

    assert not missing_output.exists()
    assert manifest_path.read_bytes() == manifest_before


@pytest.mark.parametrize("orphan_bundle", [True, False])
@pytest.mark.parametrize("swap_kind", ["metadata_and_journal", "journal_only"])
def test_reference_recovery_rejects_post_validation_transaction_swap(
    tmp_path, monkeypatch, orphan_bundle, swap_kind
):
    run_name = "swap-orphan" if orphan_bundle else "swap-record"
    run_dir = _prepare(tmp_path, run_name=run_name)
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE, extra=("--variant", "noip"))
    original_record = cli._record_stage
    if orphan_bundle:
        monkeypatch.setattr(
            cli,
            "_record_stage",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )
        with pytest.raises(RuntimeError, match="interrupted"):
            cli.main(command)
        monkeypatch.setattr(cli, "_record_stage", original_record)
    else:
        assert cli.main(command) == 0

    missing_output = run_dir / "empymod.csv"
    missing_output.unlink()
    manifest_path = run_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    original_validate = cli._validate_reference_metadata_identity
    attack = {"calls": 0}

    def validate_then_swap(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        attack["calls"] += 1
        metadata_paths = (
            run_dir / "empymod_metadata.json",
            run_dir / "artifacts" / "reference" / "empymod_metadata.json",
        )
        transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        if swap_kind == "metadata_and_journal":
            for metadata_path in metadata_paths:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["srcpts"] = 9
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            attack["root_metadata_after"] = metadata_paths[0].read_bytes()
            transaction["file_sha256"][
                "artifacts/reference/empymod_metadata.json"
            ] = cli._sha256_file(metadata_paths[1])
        else:
            transaction["inputs"]["srcpts"] = 9
        transaction_path.write_text(
            json.dumps(transaction, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(cli, "_validate_reference_metadata_identity", validate_then_swap)

    with pytest.raises(ValueError, match="hash|metadata|bundle|journal|snapshot"):
        cli.main(command)

    assert attack["calls"] == 1
    assert not missing_output.exists()
    if swap_kind == "metadata_and_journal":
        assert (run_dir / "empymod_metadata.json").read_bytes() == attack[
            "root_metadata_after"
        ]
    assert manifest_path.read_bytes() == manifest_before


def test_orphan_recovery_rechecks_journal_after_materialize_before_manifest(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="post-materialize-journal-swap")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE)
    original_record = cli._record_stage
    monkeypatch.setattr(
        cli,
        "_record_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        cli.main(command)
    monkeypatch.setattr(cli, "_record_stage", original_record)

    manifest_path = run_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    original_materialize = cli._materialize_bundle_exports
    materialized = False

    def materialize_then_swap_journal(*args, **kwargs):
        nonlocal materialized
        files = original_materialize(*args, **kwargs)
        materialized = True
        transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        transaction["inputs"]["srcpts"] = 9
        transaction_path.write_text(
            json.dumps(transaction, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return files

    monkeypatch.setattr(
        cli, "_materialize_bundle_exports", materialize_then_swap_journal
    )

    with pytest.raises(ValueError, match="journal|snapshot"):
        cli.main(command)

    assert materialized is True
    assert manifest_path.read_bytes() == manifest_before


def test_completed_stage_restores_a_missing_export_from_its_bundle(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="missing-export-run")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
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


def test_reference_srcpts_controls_real_adapter_and_resume_identity(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="reference-srcpts")
    case = cli.load_benchmark_case(LEI_CASE)
    captured = {}

    def fake_reference(times, config, *, mode, srcpts):
        captured["srcpts"] = srcpts
        captured["config_srcpts"] = config.empymod_srcpts
        return {**_fake_response(case), "reference_mode": mode}

    monkeypatch.setattr(cli, "get_empymod_reference", fake_reference)
    command = _command(
        "reference",
        run_dir,
        LEI_CASE,
        extra=("--variant", "noip", "--srcpts", "9"),
    )

    assert cli.main(command) == 0
    assert captured["srcpts"] == 9
    assert captured["config_srcpts"] == 9
    metadata = json.loads((run_dir / "empymod_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert metadata["srcpts"] == 9
    assert manifest["stages"]["reference"]["inputs"]["srcpts"] == 9
    identity = cli._load_pipeline_module()._empymod_reference_identity(
        cli._pipeline_config_for_case(case, "noip")
    )
    assert metadata["reference_identity"] == identity
    assert manifest["stages"]["reference"]["inputs"]["reference_identity"] == identity
    journal = json.loads(
        (run_dir / "artifacts" / "reference" / "_transaction.json").read_text(
            encoding="utf-8"
        )
    )
    assert journal["inputs"]["reference_identity"] == identity

    with pytest.raises(ValueError, match="inputs|resume|different"):
        cli.main(
            _command(
                "reference",
                run_dir,
                LEI_CASE,
                extra=("--variant", "noip", "--srcpts", "17"),
            )
        )


def test_reference_defaults_to_high_order_published_quadrature(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="reference-default-srcpts")
    case = cli.load_benchmark_case(LEI_CASE)
    captured = {}

    def fake_reference(times, config, *, mode, srcpts):
        captured["srcpts"] = srcpts
        captured["config_srcpts"] = config.empymod_srcpts
        return {**_fake_response(case), "reference_mode": mode}

    monkeypatch.setattr(cli, "get_empymod_reference", fake_reference)

    assert cli.main(_command("reference", run_dir, LEI_CASE)) == 0

    metadata = json.loads((run_dir / "empymod_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert captured == {"srcpts": 17, "config_srcpts": 17}
    assert metadata["srcpts"] == 17
    assert manifest["stages"]["reference"]["inputs"]["srcpts"] == 17


@pytest.mark.parametrize("srcpts", ["0", "-1"])
def test_reference_rejects_nonpositive_srcpts_without_running(tmp_path, srcpts):
    run_dir = _prepare(tmp_path, run_name=f"bad-srcpts-{srcpts}")
    with pytest.raises(SystemExit):
        cli.main(
            _command(
                "reference",
                run_dir,
                LEI_CASE,
                extra=("--variant", "noip", "--srcpts", srcpts),
            )
        )


def _remove_v1_reference_input_srcpts(run_dir):
    for path in (
        run_dir / "manifest.json",
        run_dir / "artifacts" / "reference" / "_transaction.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "manifest.json":
            payload["stages"]["reference"]["inputs"].pop("srcpts")
        else:
            payload["inputs"].pop("srcpts")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _convert_to_v1_implicit_reference_srcpts(run_dir):
    _remove_v1_reference_input_srcpts(run_dir)
    metadata_paths = (
        run_dir / "empymod_metadata.json",
        run_dir / "artifacts" / "reference" / "empymod_metadata.json",
    )
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("srcpts")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = cli._sha256_file(metadata_paths[0])
    assert cli._sha256_file(metadata_paths[1]) == digest
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["reference"]["file_sha256"]["empymod_metadata.json"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["file_sha256"]["artifacts/reference/empymod_metadata.json"] = digest
    transaction_path.write_text(
        json.dumps(transaction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_reference_metadata_srcpts(run_dir, replacement):
    metadata_paths = (
        run_dir / "empymod_metadata.json",
        run_dir / "artifacts" / "reference" / "empymod_metadata.json",
    )
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if replacement == "missing":
            payload.pop("srcpts")
        else:
            payload["srcpts"] = replacement
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = cli._sha256_file(metadata_paths[0])
    assert cli._sha256_file(metadata_paths[1]) == digest
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["reference"]["file_sha256"]["empymod_metadata.json"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["file_sha256"]["artifacts/reference/empymod_metadata.json"] = digest
    transaction_path.write_text(
        json.dumps(transaction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_reference_identity_everywhere(run_dir, replacement):
    metadata_paths = (
        run_dir / "empymod_metadata.json",
        run_dir / "artifacts" / "reference" / "empymod_metadata.json",
    )
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if replacement == "missing":
            payload.pop("reference_identity")
        else:
            payload["reference_identity"] = replacement
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = cli._sha256_file(metadata_paths[0])
    assert cli._sha256_file(metadata_paths[1]) == digest

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest["stages"]["reference"]["inputs"]
    if replacement == "missing":
        inputs.pop("reference_identity")
    else:
        inputs["reference_identity"] = replacement
    manifest["stages"]["reference"]["file_sha256"]["empymod_metadata.json"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    if replacement == "missing":
        transaction["inputs"].pop("reference_identity")
    else:
        transaction["inputs"]["reference_identity"] = replacement
    transaction["file_sha256"]["artifacts/reference/empymod_metadata.json"] = digest
    transaction_path.write_text(
        json.dumps(transaction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_reference_metadata_identity_and_hashes(run_dir, replacement):
    metadata_paths = (
        run_dir / "empymod_metadata.json",
        run_dir / "artifacts" / "reference" / "empymod_metadata.json",
    )
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["reference_identity"] = replacement
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = cli._sha256_file(metadata_paths[0])
    assert cli._sha256_file(metadata_paths[1]) == digest
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["reference"]["file_sha256"]["empymod_metadata.json"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    transaction_path = run_dir / "artifacts" / "reference" / "_transaction.json"
    transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    transaction["file_sha256"]["artifacts/reference/empymod_metadata.json"] = digest
    transaction_path.write_text(
        json.dumps(transaction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_reference_resume_rejects_rehashed_metadata_identity_tamper(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="rid-meta")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE)
    assert cli.main(command) == 0
    _rewrite_reference_metadata_identity_and_hashes(
        run_dir, {"equation": "full-wave"}
    )

    with pytest.raises(ValueError, match="identity|metadata"):
        cli.main(command)


@pytest.mark.parametrize("replacement", ["missing", {"equation": "full-wave"}])
def test_reference_resume_rejects_rehashed_reference_identity_tamper(
    tmp_path, monkeypatch, replacement
):
    run_dir = _prepare(tmp_path, run_name="reference-identity-tamper")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE)
    assert cli.main(command) == 0
    _rewrite_reference_identity_everywhere(run_dir, replacement)

    with pytest.raises(ValueError, match="identity|inputs|metadata"):
        cli.main(command)


@pytest.mark.parametrize(
    ("identity_path", "replacement"),
    [
        (("fourier_transform", "parameters", "pts_per_dec"), False),
        (("magnetic_permeability", "horizontal"), True),
    ],
    ids=["integer-to-boolean", "float-to-boolean"],
)
def test_reference_resume_rejects_synchronized_json_type_confusion(
    tmp_path, monkeypatch, identity_path, replacement
):
    run_dir = _prepare(tmp_path, run_name=f"reference-type-confusion-{replacement}")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE)
    assert cli.main(command) == 0
    replacement_identity = json.loads(
        json.dumps(cli._approved_empymod_reference_identity())
    )
    target = replacement_identity
    for key in identity_path[:-1]:
        target = target[key]
    target[identity_path[-1]] = replacement
    _rewrite_reference_identity_everywhere(run_dir, replacement_identity)

    with pytest.raises(ValueError, match="identity|inputs|metadata"):
        cli.main(command)


def test_effect_rejects_reference_transform_identity_mismatch_before_writing(
    tmp_path, monkeypatch
):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    mismatched = json.loads(
        (
            runs["ip_reference"] / "artifacts" / "reference" / "empymod_metadata.json"
        ).read_text(encoding="utf-8")
    )["reference_identity"]
    mismatched["fourier_transform"]["parameters"]["pts_per_dec"] = -1
    _rewrite_reference_identity_everywhere(runs["ip_reference"], mismatched)
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name="effect-reference-identity-mismatch",
    )
    manifest_path = effect_run / "manifest.json"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="identity|transform"):
        cli.main(_effect_args(effect_run, runs))

    assert manifest_path.read_bytes() == manifest_before
    assert not (effect_run / "effect").exists()


@pytest.mark.parametrize(
    ("identity_path", "replacement"),
    [
        (("fourier_transform", "parameters", "pts_per_dec"), False),
        (("magnetic_permeability", "horizontal"), True),
    ],
    ids=["integer-to-boolean", "float-to-boolean"],
)
def test_effect_rejects_synchronized_reference_json_type_confusion(
    tmp_path, monkeypatch, identity_path, replacement
):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    replacement_identity = json.loads(
        json.dumps(cli._approved_empymod_reference_identity())
    )
    target = replacement_identity
    for key in identity_path[:-1]:
        target = target[key]
    target[identity_path[-1]] = replacement
    _rewrite_reference_identity_everywhere(runs["ip_reference"], replacement_identity)
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name=f"effect-type-confusion-{replacement}",
    )
    manifest_path = effect_run / "manifest.json"
    manifest_before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="identity|inputs|metadata"):
        cli.main(_effect_args(effect_run, runs))

    assert manifest_path.read_bytes() == manifest_before
    assert not (effect_run / "effect").exists()


def test_effect_compares_complete_identities_after_each_source_validation(
    tmp_path, monkeypatch
):
    source_dirs = {}
    for role in ("noip_simpeg", "noip_reference", "ip_simpeg", "ip_reference"):
        source_dirs[role] = tmp_path / role
        source_dirs[role].mkdir()
    approved = cli._approved_empymod_reference_identity()
    alternate = json.loads(json.dumps(approved))
    alternate["fourier_transform"]["parameters"]["pts_per_dec"] = -1
    calls = []

    def validated_source_seam(run_dir, *, role, **_kwargs):
        calls.append(role)
        identity = {
            "run_dir": str(run_dir),
            "run_id": role,
            "case_id": "song2025_layered_pair",
            "case_hash": "0" * 64,
            "level": "S0T0B0",
            "solver_id": "empymod" if "reference" in role else SIMPEG_SOLVER,
            "stage": "reference" if "reference" in role else "simpeg",
            "variant": "ip" if role == "ip_simpeg" else "noip",
            "evidence_file": "validated.csv",
            "evidence_file_sha256": "1" * 64,
        }
        if "reference" in role:
            identity["srcpts"] = 5
            identity["reference_identity"] = (
                alternate if role == "ip_reference" else approved
            )
        return run_dir / "validated.csv", identity

    monkeypatch.setattr(cli, "_source_run_evidence", validated_source_seam)
    args = SimpleNamespace(
        noip_simpeg_run=source_dirs["noip_simpeg"],
        noip_reference_run=source_dirs["noip_reference"],
        ip_simpeg_run=source_dirs["ip_simpeg"],
        ip_reference_run=source_dirs["ip_reference"],
    )

    with pytest.raises(ValueError, match="transform identity"):
        cli._validated_effect_sources(
            args,
            effect_run_dir=tmp_path / "effect",
            effect_case=cli.load_benchmark_case(SONG_CASE),
            effect_manifest={},
        )

    assert calls == ["noip_simpeg", "noip_reference", "ip_simpeg", "ip_reference"]


@pytest.mark.parametrize("replacement", ["missing", 9])
def test_reference_resume_rejects_rehashed_metadata_srcpts_tamper(
    tmp_path, monkeypatch, replacement
):
    run_dir = _prepare(tmp_path, run_name=f"metadata-resume-{replacement}")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
    )
    command = _command("reference", run_dir, LEI_CASE)
    assert cli.main(command) == 0
    assert cli.main(command) == 0
    _rewrite_reference_metadata_srcpts(run_dir, replacement)

    with pytest.raises(ValueError, match="metadata|srcpts|identity"):
        cli.main(command)


@pytest.mark.parametrize("replacement", ["missing", 9])
def test_effect_rejects_rehashed_reference_metadata_srcpts_tamper(
    tmp_path, monkeypatch, replacement
):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    _rewrite_reference_metadata_srcpts(runs["noip_reference"], replacement)
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name=f"metadata-effect-{replacement}",
    )

    with pytest.raises(ValueError, match="metadata|srcpts|identity"):
        cli.main(_effect_args(effect_run, runs))


def test_reference_metadata_hash_and_json_use_one_byte_snapshot(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="single-read-reference")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    command = _command("reference", run_dir, LEI_CASE)
    assert cli.main(command) == 0
    metadata_path = run_dir / "artifacts" / "reference" / "empymod_metadata.json"
    original_bytes = metadata_path.read_bytes()
    changed = json.loads(original_bytes)
    changed["srcpts"] = 9
    changed_bytes = json.dumps(changed).encode("utf-8")
    original_read_bytes = Path.read_bytes
    reads = 0

    def swapping_read_bytes(path):
        nonlocal reads
        if path == metadata_path:
            reads += 1
            return original_bytes if reads == 1 else changed_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)
    assert cli.main(command) == 0
    assert reads == 1


def test_transaction_journal_hash_and_json_use_one_byte_snapshot(tmp_path, monkeypatch):
    run_dir = _prepare(tmp_path, run_name="single-read-journal")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {
            **_fake_response(case),
            "reference_mode": mode,
        },
    )
    assert cli.main(_command("reference", run_dir, LEI_CASE)) == 0
    journal_path = run_dir / "artifacts" / "reference" / "_transaction.json"
    original_bytes = journal_path.read_bytes()
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path):
        nonlocal reads
        if path == journal_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    reference_identity = cli._approved_empymod_reference_identity()
    journal, files, snapshot_sha256 = cli._verify_transaction_bundle(
        run_dir,
        "reference",
        {
            "variant": "noip",
            "srcpts": 17,
            "reference_identity": reference_identity,
        },
    )

    assert reads == 1
    assert journal["inputs"] == {
        "srcpts": 17,
        "variant": "noip",
        "reference_identity": reference_identity,
    }
    assert files
    assert snapshot_sha256 == cli._sha256_bytes(original_bytes)


def test_effect_rejects_mismatched_reference_srcpts_before_writing(
    tmp_path, monkeypatch
):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    ip_reference_9 = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="empymod",
        run_name="ip-reference-9",
    )
    assert cli.main(
        _command(
            "reference",
            ip_reference_9,
            SONG_CASE,
            extra=("--variant", "cole-cole-exact", "--srcpts", "9"),
        )
    ) == 0
    runs["ip_reference"] = ip_reference_9
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name="effect-srcpts-mismatch",
    )
    manifest_path = effect_run / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    monkeypatch.setattr(
        cli,
        "write_polarization_effect_artifacts",
        lambda *args, **kwargs: pytest.fail(
            "mismatched reference quadrature must fail before effect API"
        ),
    )

    with pytest.raises(ValueError, match="srcpts|quadrature"):
        cli.main(_effect_args(effect_run, runs))

    assert manifest_path.read_bytes() == manifest_before
    assert not (effect_run / "effect").exists()


def test_reference_rejects_v1_stage_with_implicit_default_srcpts(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="legacy-v1-reference")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
    )
    assert cli.main(_command("reference", run_dir, LEI_CASE)) == 0
    _convert_to_v1_implicit_reference_srcpts(run_dir)
    with pytest.raises(ValueError, match="inputs|srcpts"):
        cli.main(_command("reference", run_dir, LEI_CASE))


def test_effect_rejects_v1_reference_runs_with_implicit_default_srcpts(
    tmp_path, monkeypatch
):
    runs = _build_effect_source_runs(tmp_path, monkeypatch)
    _convert_to_v1_implicit_reference_srcpts(runs["noip_reference"])
    _convert_to_v1_implicit_reference_srcpts(runs["ip_reference"])
    effect_run = _prepare(
        tmp_path,
        case=SONG_CASE,
        solver="polarization_effect",
        run_name="legacy-v1-effect",
    )

    with pytest.raises(ValueError, match="srcpts|inputs"):
        cli.main(_effect_args(effect_run, runs))


def test_reference_rejects_missing_srcpts_even_when_all_metadata_hashes_are_recomputed(
    tmp_path, monkeypatch
):
    run_dir = _prepare(tmp_path, run_name="tampered-new-reference")
    case = cli.load_benchmark_case(LEI_CASE)
    monkeypatch.setattr(
        cli,
        "get_empymod_reference",
        lambda times, config, *, mode, srcpts: {**_fake_response(case), "reference_mode": mode},
    )
    assert cli.main(
        _command("reference", run_dir, LEI_CASE, extra=("--srcpts", "17"))
    ) == 0
    _convert_to_v1_implicit_reference_srcpts(run_dir)

    with pytest.raises(ValueError, match="srcpts|metadata|inputs"):
        cli.main(_command("reference", run_dir, LEI_CASE))


def test_pyproject_registers_sotem_validation_entrypoint():
    payload = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'atem3d-sotem-validate = "atem3d.sotem_validation_cli:main"' in payload
