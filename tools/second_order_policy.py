#!/usr/bin/env python3
"""Enforce the production DOLFINx SOTEM pipeline's second-order-only policy.

The production finite-element solver uses N1curl order 2.  First-order
Nedelec elements remain only in explicitly named legacy/reference files and
must not be selectable through the production configuration or benchmark
wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "dolfinx" / "sotem_pipeline.py"
BENCHMARK_WRAPPER = ROOT / "dolfinx" / "run_sotem_benchmark.py"
README = ROOT / "README.md"
POLICY_MARKER = "## 13. 二阶 Nédélec 生产策略"


def _replace_exact(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one occurrence, found {count}"
        )
    return text.replace(old, new, 1)


def _apply_pipeline(text: str) -> str:
    if "REQUIRED_NEDELEC_ORDER = 2" not in text:
        text = _replace_exact(
            text,
            "LOCAL_MESH_MAX_ASPECT = 100.0\n",
            "LOCAL_MESH_MAX_ASPECT = 100.0\nREQUIRED_NEDELEC_ORDER = 2\n",
            label="insert required order constant",
        )

    if "nedelec_order: int = 1" in text:
        text = _replace_exact(
            text,
            "    nedelec_order: int = 1\n",
            "    nedelec_order: int = REQUIRED_NEDELEC_ORDER\n",
            label="replace PipelineConfig default",
        )

    old_validation = (
        "    nedelec_order = int(config.nedelec_order)\n"
        "    if nedelec_order not in {1, 2}:\n"
        "        raise ValueError(f\"nedelec_order must be 1 or 2; got {nedelec_order}\")\n"
    )
    new_validation = (
        "    nedelec_order = int(config.nedelec_order)\n"
        "    if nedelec_order != REQUIRED_NEDELEC_ORDER:\n"
        "        raise ValueError(\n"
        "            f\"nedelec_order is fixed to {REQUIRED_NEDELEC_ORDER}; \"\n"
        "            f\"got {nedelec_order}\"\n"
        "        )\n"
    )
    if old_validation in text:
        text = _replace_exact(
            text,
            old_validation,
            new_validation,
            label="replace order validation",
        )

    old_fallback = (
        "    nedelec_order = int(config.nedelec_order) if config is not None else 1\n"
    )
    new_fallback = (
        "    nedelec_order = (\n"
        "        int(config.nedelec_order)\n"
        "        if config is not None\n"
        "        else REQUIRED_NEDELEC_ORDER\n"
        "    )\n"
        "    if nedelec_order != REQUIRED_NEDELEC_ORDER:\n"
        "        raise ValueError(\n"
        "            f\"production function spaces require N1curl order \"\n"
        "            f\"{REQUIRED_NEDELEC_ORDER}; got {nedelec_order}\"\n"
        "        )\n"
    )
    if old_fallback in text:
        text = _replace_exact(
            text,
            old_fallback,
            new_fallback,
            label="replace function-space fallback",
        )

    return text


def _apply_benchmark_wrapper(text: str) -> str:
    if '"--nedelec-order=2",' not in text:
        text = _replace_exact(
            text,
            '        "--magnetic-dbdt-mode=curl",\n',
            '        "--magnetic-dbdt-mode=curl",\n'
            '        "--nedelec-order=2",\n',
            label="make benchmark order explicit",
        )
    return text


def _replace_explicit_first_order_tokens() -> list[Path]:
    changed: list[Path] = []
    roots = [ROOT / "benchmarks", ROOT / "examples", ROOT / "dolfinx"]
    suffixes = {".yaml", ".yml", ".json", ".sh", ".md"}
    replacements = (
        ("--nedelec-order=1", "--nedelec-order=2"),
        ("--nedelec-order 1", "--nedelec-order 2"),
        ("nedelec_order: 1", "nedelec_order: 2"),
        ("nedelec_order=1", "nedelec_order=2"),
    )
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.name == "legacy_total_field_baseline.py":
                continue
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, new in replacements:
                updated = updated.replace(old, new)
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                changed.append(path)
    return changed


def _apply_readme(text: str) -> str:
    if POLICY_MARKER in text:
        return text
    appendix = """
    if not text.endswith("\n"):
        appendix += "\n"
    appendix += (
        "\n## 13. 二阶 Nédélec 生产策略\n\n"
        "DOLFINx 生产正演固定使用二阶 `N1curl(2)` 边元。"
        "`PipelineConfig.nedelec_order` 的默认值为 2，且生产配置若传入"
        "其他阶次会立即报错。公开基准包装器也显式传递"
        "`--nedelec-order=2`。一阶结果仅允许保留在明确标注的"
        "legacy/reference 文件中，不得作为正式坝体弱异常结论。\n"
    )
    return text + appendix


def _write_if_changed(path: Path, updated: str) -> bool:
    original = path.read_text(encoding="utf-8")
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def apply_policy() -> list[Path]:
    changed: list[Path] = []
    pipeline_text = PIPELINE.read_text(encoding="utf-8")
    if _write_if_changed(PIPELINE, _apply_pipeline(pipeline_text)):
        changed.append(PIPELINE)

    wrapper_text = BENCHMARK_WRAPPER.read_text(encoding="utf-8")
    if _write_if_changed(
        BENCHMARK_WRAPPER,
        _apply_benchmark_wrapper(wrapper_text),
    ):
        changed.append(BENCHMARK_WRAPPER)

    readme_text = README.read_text(encoding="utf-8")
    if _write_if_changed(README, _apply_readme(readme_text)):
        changed.append(README)

    changed.extend(_replace_explicit_first_order_tokens())
    check_policy()
    return sorted(set(changed))


def check_policy() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")
    wrapper = BENCHMARK_WRAPPER.read_text(encoding="utf-8")

    required = {
        "required order constant": "REQUIRED_NEDELEC_ORDER = 2",
        "second-order dataclass default": (
            "nedelec_order: int = REQUIRED_NEDELEC_ORDER"
        ),
        "strict production validation": (
            "if nedelec_order != REQUIRED_NEDELEC_ORDER:"
        ),
        "second-order function-space fallback": (
            "else REQUIRED_NEDELEC_ORDER"
        ),
    }
    missing = [label for label, token in required.items() if token not in pipeline]
    if missing:
        raise RuntimeError("second-order policy is incomplete: " + ", ".join(missing))

    forbidden_pipeline = {
        "first-order dataclass default": "nedelec_order: int = 1",
        "mixed first/second validation": "nedelec_order not in {1, 2}",
        "first-order function-space fallback": (
            "if config is not None else 1"
        ),
    }
    present = [
        label for label, token in forbidden_pipeline.items() if token in pipeline
    ]
    if present:
        raise RuntimeError(
            "first-order production path remains: " + ", ".join(present)
        )

    if '"--nedelec-order=2",' not in wrapper:
        raise RuntimeError("benchmark wrapper does not explicitly request order 2")

    forbidden_patterns = (
        re.compile(r"--nedelec-order(?:=|\s+)1(?:\D|$)"),
        re.compile(r"\bnedelec_order\s*:\s*1\b"),
        re.compile(r"\bnedelec_order\s*=\s*1\b"),
    )
    offenders: list[str] = []
    scan_roots = [ROOT / "benchmarks", ROOT / "examples"]
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(pattern.search(text) for pattern in forbidden_patterns):
                offenders.append(str(path.relative_to(ROOT)))
    if offenders:
        raise RuntimeError(
            "explicit first-order production examples remain: "
            + ", ".join(sorted(offenders))
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.apply:
        changed = apply_policy()
        for path in changed:
            print(path.relative_to(ROOT))
    else:
        check_policy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
