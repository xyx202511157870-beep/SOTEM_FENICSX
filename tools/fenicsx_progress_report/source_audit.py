"""Audit the report source for evidence coverage and claim boundaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Sequence
from typing import Any, Mapping


REQUIRED_SECTION_TITLES = (
    "摘要与当前状态",
    "项目起点与目标",
    "坐标、几何与物理约定",
    "控制方程与离散方法",
    "源项与电极调试",
    "网格与计算域调试",
    "接收点与场量恢复",
    "初始场与主场—二次场演进",
    "时间离散与关断调试",
    "Cole-Cole/Debye 扩展",
    "均匀半空间三方验证",
    "分层模型与第一阶段收敛研究",
    "第二阶段论文基线方案",
    "复杂模型应用与边界",
    "软件结构、测试与复现",
    "当前评价与下一步",
    "附录 A：调试时间线",
    "附录 B：调试效果图谱",
    "附录 C：模型参数与误差表",
    "附录 D：运行命令与产物索引",
    "附录 E：验证状态矩阵",
)


@dataclass(frozen=True)
class SourceAudit:
    codes: list[str]
    missing_sections: list[str]
    unknown_figure_ids: list[str]
    missing_body_figure_ids: list[str]

    @property
    def ok(self) -> bool:
        return not self.codes


def _headings(source: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"^##\s+(.+?)\s*$", source, flags=re.MULTILINE):
        heading = re.sub(r"^\d+\.\s*", "", match.group(1)).strip()
        values.append(heading)
    return values


def _has_all(source: str, tokens: tuple[str, ...]) -> bool:
    return all(token in source for token in tokens)


def audit_source(source: str, selection: Mapping[str, Any]) -> SourceAudit:
    codes: list[str] = []
    headings = _headings(source)
    missing_sections = [
        title for title in REQUIRED_SECTION_TITLES if title not in headings
    ]
    if missing_sections:
        codes.append("missing_required_section")

    figure_ids = re.findall(r"\[\[FIGURE:([A-Za-z0-9_-]+)\]\]", source)
    known_ids = {
        str(figure.get("id", "")) for figure in selection.get("figures", [])
    }
    unknown_figure_ids = sorted(set(figure_ids) - known_ids)
    if unknown_figure_ids:
        codes.extend(f"unknown_figure_id:{figure_id}" for figure_id in unknown_figure_ids)
    body_ids = {
        str(figure.get("id", ""))
        for figure in selection.get("figures", [])
        if figure.get("placement") == "body"
    }
    missing_body_figure_ids = sorted(body_ids - set(figure_ids))
    if missing_body_figure_ids:
        codes.extend(
            f"missing_body_figure:{figure_id}"
            for figure_id in missing_body_figure_ids
        )
    has_appendix_figures = any(
        figure.get("placement") == "appendix"
        for figure in selection.get("figures", [])
    )
    if has_appendix_figures and "[[FIGURE_ATLAS]]" not in source:
        codes.append("missing_figure_atlas")

    required_claims = {
        "missing_coordinate_convention": ("地下为正", "空气为负", "z=0"),
        "missing_authoritative_paths": ("src/atem3d/", "dolfinx/sotem_pipeline.py"),
        "missing_uniform_metrics": ("2.75%", "3.13%", "5.97%"),
        "missing_layered_metrics": ("0.83%", "2.76%", "9.22%"),
        "missing_first_stage_axis_results": (
            "网格轴通过",
            "时间步轴未通过",
            "计算域轴未通过",
        ),
        "missing_stage2_pending_boundary": (
            "第二阶段",
            "长时间求解尚未完成",
            "待完成",
        ),
        "missing_status_taxonomy": (
            "已验证",
            "通过",
            "未通过",
            "待完成",
            "诊断用途",
            "历史实现",
        ),
    }
    for code, tokens in required_claims.items():
        if not _has_all(source, tokens):
            codes.append(code)

    for token in ("TBD", "TODO", "待定", "占位"):
        if token in source:
            codes.append(f"placeholder_token:{token}")
    overclaim_patterns = (
        r"已经验证任意三维异常体",
        r"已经验证.*复杂地形",
        r"可直接验证.*坝体",
    )
    if any(re.search(pattern, source) for pattern in overclaim_patterns):
        codes.append("overclaim:universal_3d_validation")

    return SourceAudit(
        codes=codes,
        missing_sections=missing_sections,
        unknown_figure_ids=unknown_figure_ids,
        missing_body_figure_ids=missing_body_figure_ids,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = args.source.read_text(encoding="utf-8")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = audit_source(source, selection)
    print(
        json.dumps(
            {
                "ok": report.ok,
                "codes": report.codes,
                "missing_sections": report.missing_sections,
                "unknown_figure_ids": report.unknown_figure_ids,
                "missing_body_figure_ids": report.missing_body_figure_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
