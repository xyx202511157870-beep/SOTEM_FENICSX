from __future__ import annotations

from pathlib import Path

from atem3d.seepage_case_matrix import build_case_matrix
from tools.run_atem3d_with_runtime import runtime_audit_path
from tools.run_seepage_verification_matrix import build_case_command


def test_simpeg_case_command_uses_selected_preflight_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    selected = str(tmp_path / "runtime" / "python.exe")
    monkeypatch.setenv("ATEM3D_SIMPEG_PYTHON", selected)
    case = next(item for item in build_case_matrix() if item.solver == "simpeg")

    command = build_case_command(case, tmp_path)

    assert command[0] == selected
    assert Path(command[1]).name == "run_atem3d_with_runtime.py"
    assert command[2] == "run"


def test_runtime_audit_is_written_beside_solver_output(tmp_path: Path) -> None:
    result = tmp_path / "case" / "result.h5"
    assert runtime_audit_path(["run", "case.yaml", "--output", str(result)]) == (
        result.parent / "runtime_environment.json"
    )
    assert runtime_audit_path(["--preflight-pardiso"]) is None
