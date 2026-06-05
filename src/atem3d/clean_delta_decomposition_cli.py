"""CLI for clean-delta/no-IP baseline error decomposition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clean_delta_decomposition import (
    clean_delta_decomposition_report,
    load_sampled_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose a corrected IP sampled report into no-IP baseline error "
            "and clean-delta correction mismatch."
        )
    )
    parser.add_argument("raw_ip_report", type=Path)
    parser.add_argument("corrected_ip_report", type=Path)
    parser.add_argument("noip_report", type=Path)
    parser.add_argument(
        "--components",
        default=None,
        help="Comma-separated sampled component names. Overrides --component-prefix.",
    )
    parser.add_argument(
        "--component-prefix",
        default=None,
        help="Select sampled components by prefix, e.g. Hz. Defaults to all samples.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args(argv)

    component_names = _parse_components(args.components)
    raw_ip = load_sampled_report(
        args.raw_ip_report,
        component_names=component_names,
        component_prefix=args.component_prefix,
    )
    corrected_ip = load_sampled_report(
        args.corrected_ip_report,
        component_names=raw_ip.names,
    )
    noip = load_sampled_report(
        args.noip_report,
        component_names=raw_ip.names,
    )

    report = clean_delta_decomposition_report(raw_ip, corrected_ip, noip)
    report["inputs"] = {
        "raw_ip_report": str(args.raw_ip_report),
        "corrected_ip_report": str(args.corrected_ip_report),
        "noip_report": str(args.noip_report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


def _parse_components(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("--components must contain at least one component name")
    return names


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
