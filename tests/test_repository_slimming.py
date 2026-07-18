from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_removed_surfaces_are_absent():
    forbidden = [
        "COMSOL",
        "sotem_ip",
        "dolfinx/current_task_runs",
        "src/atem3d/comsol_seepage_channel_3d.py",
        "src/atem3d/four_way_validation.py",
        "tools/run_comsol_seepage_channel_3d.py",
    ]

    assert [path for path in forbidden if (ROOT / path).exists()] == []


def test_active_sources_have_no_comsol_references():
    active_roots = [ROOT / "src", ROOT / "dolfinx", ROOT / "tools", ROOT / "tests"]
    offenders = []
    for active_root in active_roots:
        for path in active_root.rglob("*"):
            if path.suffix not in {".py", ".sh"}:
                continue
            if "repository_slimming" in path.name:
                continue
            if "comsol" in path.read_text(encoding="utf-8").casefold():
                offenders.append(path.relative_to(ROOT).as_posix())

    assert offenders == []


def test_three_solver_surfaces_and_known_issue_are_retained():
    required = [
        "src/atem3d/seepage_reference.py",
        "src/atem3d/empymod_compare.py",
        "dolfinx/seepage_channel_full_domain.py",
        "dolfinx/symmetric_full_domain_mesh.py",
        "tools/run_seepage_channel_benchmark.py",
        "examples/seepage_channel_100m_5rx_simpeg_channel.yaml",
    ]

    assert [path for path in required if not (ROOT / path).is_file()] == []
    status = (ROOT / "docs/current_status.md").read_text(encoding="utf-8")
    assert "晚期 Ex" in status
    assert "未通过空间收敛验证" in status
