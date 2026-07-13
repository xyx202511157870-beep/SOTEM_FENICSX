"""Deterministic evidence inventory for the FEniCSx progress report."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_EXCLUDED_PARTS = (".git", ".worktrees", "__pycache__", "tmp")


@dataclass(frozen=True)
class ImageEvidence:
    path: str
    suffix: str
    byte_size: int
    width: int
    height: int
    modified_utc: str
    sha256: str
    duplicate_of: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(
    root: Path,
    *,
    excluded_parts: tuple[str, ...] = DEFAULT_EXCLUDED_PARTS,
) -> list[ImageEvidence]:
    root = root.resolve()
    records: list[ImageEvidence] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix().lower()):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in excluded_parts for part in relative.parts):
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        with Image.open(path) as image:
            width, height = image.size
        stat = path.stat()
        records.append(
            ImageEvidence(
                path=relative.as_posix(),
                suffix=suffix,
                byte_size=stat.st_size,
                width=width,
                height=height,
                modified_utc=datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                sha256=_sha256(path),
            )
        )
    return records


def deduplicate_images(records: Sequence[ImageEvidence]) -> list[ImageEvidence]:
    canonical_by_hash: dict[str, str] = {}
    result: list[ImageEvidence] = []
    for record in records:
        canonical = canonical_by_hash.setdefault(record.sha256, record.path)
        result.append(
            replace(
                record,
                duplicate_of=None if canonical == record.path else canonical,
            )
        )
    return result


def write_inventory(
    records: Sequence[ImageEvidence], json_path: Path, csv_path: Path
) -> None:
    rows = [asdict(record) for record in records]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"images": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(asdict(ImageEvidence("", "", 0, 0, 0, "", "")))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_git_timeline(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line:
            continue
        commit, date, subject = line.split("\t", 2)
        events.append({"commit": commit, "date": date, "subject": subject})
    return events


def extract_git_timeline(root: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "log", "--date=iso-strict", "--pretty=format:%H%x09%ad%x09%s"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return parse_git_timeline(completed.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--timeline", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = deduplicate_images(discover_images(args.root))
    write_inventory(records, args.json, args.csv)
    timeline = extract_git_timeline(args.root)
    args.timeline.parent.mkdir(parents=True, exist_ok=True)
    args.timeline.write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    duplicate_count = sum(record.duplicate_of is not None for record in records)
    print(
        json.dumps(
            {
                "discovered": len(records),
                "canonical": len(records) - duplicate_count,
                "duplicates": duplicate_count,
                "git_events": len(timeline),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
