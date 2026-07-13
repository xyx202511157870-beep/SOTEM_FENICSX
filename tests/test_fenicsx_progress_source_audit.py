import json
from pathlib import Path

from tools.fenicsx_progress_report.source_audit import (
    REQUIRED_SECTION_TITLES,
    audit_source,
    main,
)


def _complete_source(extra: str = "") -> str:
    sections = "\n".join(f"## {title}\n正文。" for title in REQUIRED_SECTION_TITLES)
    return "\n".join(
        [
            "# FEniCSx 算法建立与调试进展",
            sections,
            "地下为正，空气为负；地面 z=0。",
            "当前权威实现位于 src/atem3d/ 与 dolfinx/sotem_pipeline.py。",
            "均匀半空间 dBz/dt 中位误差约 2.75%，RMS 约 3.13%，最大约 5.97%。",
            "12 km 分层模型中位误差约 0.83%，RMS 约 2.76%，最大约 9.22%。",
            "第一阶段网格轴通过；时间步轴未通过；计算域轴未通过。",
            "第二阶段长时间求解尚未完成，状态为待完成。",
            "已验证；通过；未通过；待完成；诊断用途；历史实现。",
            extra,
        ]
    )


def test_audit_reports_missing_sections_and_unknown_figures() -> None:
    source = "# 报告\n均匀半空间中位误差约 2.75%。\n[[FIGURE:missing]]"
    selection = {"figures": [{"id": "known", "placement": "body"}]}

    report = audit_source(source, selection)

    assert "missing_required_section" in report.codes
    assert report.unknown_figure_ids == ["missing"]
    assert report.missing_body_figure_ids == ["known"]


def test_audit_accepts_required_claim_boundaries_and_body_figure() -> None:
    source = _complete_source("[[FIGURE:known]]\n[[FIGURE_ATLAS]]")
    selection = {
        "figures": [
            {"id": "known", "placement": "body"},
            {"id": "appendix_only", "placement": "appendix"},
        ]
    }

    report = audit_source(source, selection)

    assert report.ok
    assert not report.unknown_figure_ids
    assert not report.missing_body_figure_ids


def test_audit_rejects_placeholders_and_universal_validation_claims() -> None:
    source = _complete_source(
        "TODO\n本算法已经验证任意三维异常体和复杂地形。"
    )

    report = audit_source(source, {"figures": []})

    assert "placeholder_token:TODO" in report.codes
    assert "overclaim:universal_3d_validation" in report.codes


def test_audit_requires_atlas_when_appendix_figures_exist() -> None:
    source = _complete_source()
    selection = {"figures": [{"id": "appendix_only", "placement": "appendix"}]}

    report = audit_source(source, selection)

    assert "missing_figure_atlas" in report.codes


def test_source_audit_cli_returns_success_for_complete_source(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "report.md"
    source_path.write_text(_complete_source(), encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps({"figures": []}, ensure_ascii=False), encoding="utf-8"
    )

    assert (
        main(
            [
                "--source",
                str(source_path),
                "--selection",
                str(selection_path),
            ]
        )
        == 0
    )
