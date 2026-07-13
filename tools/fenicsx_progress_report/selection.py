"""Classify, review, and audit report figure selections."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps


PHASE_IDS = (
    "legacy_simpeg",
    "formulation",
    "source_electrodes",
    "mesh_domain",
    "receiver_fields",
    "initial_primary_secondary",
    "time_turnoff",
    "cole_cole_debye",
    "uniform_three_way",
    "layered_convergence",
    "stage2_pending",
    "complex_models",
    "reproducibility",
)

ALLOWED_STATUSES = (
    "已验证",
    "通过",
    "未通过",
    "待完成",
    "诊断用途",
    "历史实现",
)


@dataclass(frozen=True)
class SelectionAudit:
    missing_paths: list[str]
    duplicate_selections: list[str]
    invalid_statuses: list[str]
    unclassified: list[str]
    uncovered: list[str]
    missing_exclusion_reasons: list[str]

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.missing_paths,
                self.duplicate_selections,
                self.invalid_statuses,
                self.unclassified,
                self.uncovered,
                self.missing_exclusion_reasons,
            )
        )


def classify_phase(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    if normalized.startswith("simpeg3d有限体积/"):
        return "legacy_simpeg"
    if normalized.startswith("dolfinx/dam_leakage_model/"):
        return "complex_models"
    if "/stage2" in normalized or "stage2_" in normalized:
        return "stage2_pending"
    if normalized.startswith("output/publication_validation/"):
        return "layered_convergence"
    if normalized.startswith("comsol/"):
        if "three_way" in normalized or "verification" in name or "comparison" in name or "error_curves" in name:
            return "uniform_three_way"
        if "source_receiver" in name or "electrode" in name:
            return "source_electrodes"
        return "mesh_domain"
    if normalized.startswith("output/model_figures/"):
        return "source_electrodes"
    if normalized.startswith("dolfinx/paper_reproduction/xue2022"):
        return "receiver_fields"
    if normalized.startswith("dolfinx/paper_reproduction/song2025"):
        if "/digitized/" in normalized or "ip_" in normalized or "debye" in normalized:
            return "cole_cole_debye"
        return "receiver_fields"
    return "unclassified"


def load_selection(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    figures = payload.get("figures", [])
    ids = [figure.get("id") for figure in figures]
    duplicate_ids = sorted({figure_id for figure_id in ids if ids.count(figure_id) > 1})
    if duplicate_ids:
        raise ValueError(f"duplicate figure id: {', '.join(duplicate_ids)}")
    return payload


def audit_selection(
    inventory: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> SelectionAudit:
    inventory_by_path = {str(record["path"]): record for record in inventory}
    figures = list(manifest.get("figures", []))
    excluded = list(manifest.get("excluded", []))
    selected_paths = {str(figure.get("path", "")) for figure in figures}
    excluded_paths = {str(item.get("path", "")) for item in excluded}
    canonical_paths = {
        path
        for path, record in inventory_by_path.items()
        if not record.get("duplicate_of")
    }
    missing_paths = sorted(path for path in selected_paths if path not in inventory_by_path)
    duplicate_selections = sorted(
        path
        for path in selected_paths
        if path in inventory_by_path and inventory_by_path[path].get("duplicate_of")
    )
    invalid_statuses = sorted(
        f"{figure.get('id', '')}:{figure.get('status', '')}"
        for figure in figures
        if figure.get("status") not in ALLOWED_STATUSES
    )
    unclassified = sorted(
        str(figure.get("id", ""))
        for figure in figures
        if figure.get("phase") not in PHASE_IDS
    )
    missing_exclusion_reasons = sorted(
        str(item.get("path", "")) for item in excluded if not str(item.get("reason", "")).strip()
    )
    uncovered = sorted(canonical_paths - selected_paths - excluded_paths)
    return SelectionAudit(
        missing_paths=missing_paths,
        duplicate_selections=duplicate_selections,
        invalid_statuses=invalid_statuses,
        unclassified=unclassified,
        uncovered=uncovered,
        missing_exclusion_reasons=missing_exclusion_reasons,
    )


def _exclude_reason(path: str) -> str | None:
    normalized = path.replace("\\", "/").lower()
    name = Path(normalized).name
    if "/pdf_pages/" in normalized:
        return "第三方论文整页截图，仅用于内部核对，不作为本项目调试产物收录。"
    if "/digitized/" in normalized and (
        "_crop" in name or "_zoom" in name or "page05" in name
    ):
        return "第三方论文直接裁剪图，仅保留项目生成的数字化曲线或叠加结果。"
    if name.endswith("_display.png"):
        return "与同组结果仅显示样式不同，不提供独立数值证据。"
    return None


def _placement(path: str) -> str:
    normalized = path.replace("\\", "/").lower()
    body_patterns = (
        "three_way_final",
        "fenicsx_domain600_forward_model",
        "fenicsx_source_receiver_zoom",
        "output/model_figures/",
        "convergence_curves.png",
        "convergence_differences.png",
        "latest_y200_dolfinx_empymod",
        "paper_y200_center_tmax1_",
        "original_dam_half_mirror_three_case_",
        "combined_ip_noip_",
        "dam_leakage_ip_ab_electrode_model_schematic",
        "dam_leakage_ip_material_markers_and_mesh",
    )
    return "body" if any(pattern in normalized for pattern in body_patterns) else "appendix"


def _status(path: str, phase: str) -> str:
    normalized = path.replace("\\", "/").lower()
    if phase == "legacy_simpeg":
        return "历史实现"
    if phase == "stage2_pending":
        return "待完成"
    if phase == "uniform_three_way" and (
        "three_way_final" in normalized
        or "fenicsx_uniform_halfspace_domain6000_far750_dt25us_full" in normalized
    ):
        return "已验证"
    if phase == "layered_convergence" and "convergence_curves.png" in normalized:
        return "通过"
    if phase == "layered_convergence" and "convergence_differences.png" in normalized:
        return "未通过"
    return "诊断用途"


def _caption(path: str, phase: str) -> str:
    name = Path(path).stem.replace("_", " ")
    lowered = name.lower()
    if "schematic" in lowered or "model" in lowered:
        kind = "模型、几何或网格示意"
    elif "error" in lowered or "difference" in lowered:
        kind = "误差与差异诊断"
    elif "convergence" in lowered or "richardson" in lowered:
        kind = "收敛性诊断"
    elif "response" in lowered or "comparison" in lowered:
        kind = "响应对比"
    elif "verification" in lowered:
        kind = "阶段验证结果"
    else:
        kind = "调试结果"
    return f"{kind}：{name}（阶段：{phase}）。"


def _figure_id(path: str, phase: str, sha256: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", Path(path).stem.lower()).strip("_")
    if not stem:
        stem = "figure"
    return f"{phase}_{stem[:48]}_{sha256[:8]}"


def bootstrap_selection(
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    figures: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for record in sorted(inventory, key=lambda item: str(item["path"])):
        if record.get("duplicate_of"):
            continue
        path = str(record["path"])
        reason = _exclude_reason(path)
        if reason:
            excluded.append({"path": path, "reason": reason})
            continue
        phase = classify_phase(path)
        if phase == "unclassified":
            phase = "reproducibility"
        status = _status(path, phase)
        figures.append(
            {
                "id": _figure_id(path, phase, str(record.get("sha256", "00000000"))),
                "path": path,
                "phase": phase,
                "status": status,
                "placement": _placement(path),
                "caption": _caption(path, phase),
                "problem": "用于记录该阶段出现的幅值、符号、几何、网格、时间离散或参考解一致性问题。",
                "change": "依据目录和运行标识对应的参数、几何或数值方案进行调整，并保留该轮输出。",
                "result": "该图作为阶段性证据；最终判断以图中曲线、相邻元数据和正文给出的误差指标为准。",
                "source_run": str(Path(path).parent).replace("\\", "/"),
            }
        )
    return {"figures": figures, "excluded": excluded}


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _short_label(path: str, limit: int = 64) -> str:
    return path if len(path) <= limit else f"...{path[-(limit - 3):]}"


def write_contact_sheets(
    records: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    output_dir: Path,
    columns: int = 4,
    thumb_size: tuple[int, int] = (360, 240),
    rows_per_sheet: int = 4,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        phase = str(record.get("phase") or classify_phase(str(record["path"])))
        grouped.setdefault(phase, []).append(record)

    margin = 20
    label_height = 42
    cell_width = thumb_size[0] + margin
    cell_height = thumb_size[1] + label_height + margin
    font = _font(14)
    outputs: list[Path] = []
    per_sheet = columns * rows_per_sheet
    for phase in sorted(grouped):
        phase_records = sorted(grouped[phase], key=lambda item: str(item["path"]))
        for sheet_index, start in enumerate(range(0, len(phase_records), per_sheet), 1):
            page_records = phase_records[start : start + per_sheet]
            row_count = (len(page_records) + columns - 1) // columns
            canvas = Image.new(
                "RGB",
                (columns * cell_width + margin, row_count * cell_height + margin),
                "#eef2f4",
            )
            draw = ImageDraw.Draw(canvas)
            for index, record in enumerate(page_records):
                row, column = divmod(index, columns)
                left = margin + column * cell_width
                top = margin + row * cell_height
                image_path = root / str(record["path"])
                with Image.open(image_path) as source:
                    preview = ImageOps.contain(source.convert("RGB"), thumb_size)
                paste_left = left + (thumb_size[0] - preview.width) // 2
                paste_top = top + (thumb_size[1] - preview.height) // 2
                canvas.paste(preview, (paste_left, paste_top))
                draw.rectangle(
                    (left, top, left + thumb_size[0], top + thumb_size[1]),
                    outline="#9aabb5",
                    width=1,
                )
                label = _short_label(str(record["path"]))
                draw.text(
                    (left, top + thumb_size[1] + 6),
                    label,
                    fill="#203a49",
                    font=font,
                )
            output = output_dir / f"{phase}-{sheet_index:03d}.png"
            canvas.save(output)
            outputs.append(output)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--contact-sheet-dir", type=Path)
    parser.add_argument("--bootstrap-selection", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.inventory.read_text(encoding="utf-8"))
    inventory = list(payload.get("images", []))
    if args.bootstrap_selection is not None:
        manifest = bootstrap_selection(inventory)
        args.bootstrap_selection.parent.mkdir(parents=True, exist_ok=True)
        args.bootstrap_selection.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "figures": len(manifest["figures"]),
                    "excluded": len(manifest["excluded"]),
                    "output": str(args.bootstrap_selection),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.audit_only:
        if args.selection is None:
            raise ValueError("--selection is required with --audit-only")
        report = audit_selection(inventory, load_selection(args.selection))
        print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
        return 0 if report.ok else 1
    if args.contact_sheet_dir is None:
        raise ValueError("--contact-sheet-dir is required unless --audit-only is used")
    canonical = [record for record in inventory if not record.get("duplicate_of")]
    records = [dict(record, phase=classify_phase(str(record["path"]))) for record in canonical]
    outputs = write_contact_sheets(
        records,
        root=args.root,
        output_dir=args.contact_sheet_dir,
    )
    phase_counts: dict[str, int] = {}
    for record in records:
        phase = str(record["phase"])
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    print(
        json.dumps(
            {
                "canonical": len(canonical),
                "contact_sheets": len(outputs),
                "phase_counts": phase_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
