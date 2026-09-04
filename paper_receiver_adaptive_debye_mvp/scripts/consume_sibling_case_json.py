#!/usr/bin/env python3
"""Copy sibling-branch case JSON into this workspace without overwriting disks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PREFIXES = (
    "generated/receiver_adaptive_debye_mvp/flow3_selector/case_",
    "generated/receiver_adaptive_debye_mvp/flow4_independent_test/case_",
)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(REPO), text=True, capture_output=True)


def _fetch() -> None:
    delay = 4
    for _attempt in range(4):
        result = _run(["git", "fetch", "origin"])
        if result.returncode == 0:
            return
        subprocess.run(["sleep", str(delay)], check=False)
        delay *= 2


def _payload_rank(payload: dict) -> int:
    if not payload.get("tasks"):
        return 0
    if payload.get("point_only") is False:
        return 2
    return 1


def _load(raw: bytes) -> dict | None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("case_id"):
        return None
    return payload


def main() -> int:
    _fetch()
    refs = _run(["git", "branch", "-r"])
    if refs.returncode != 0:
        print(refs.stderr, file=sys.stderr)
        return 1
    copied = []
    skipped = []
    for line in refs.stdout.splitlines():
        ref = line.strip()
        if not ref or "->" in ref:
            continue
        listed = _run(["git", "ls-tree", "-r", "--name-only", ref])
        if listed.returncode != 0:
            continue
        for path in listed.stdout.splitlines():
            if not path.endswith(".json") or not any(path.startswith(prefix) for prefix in PREFIXES):
                continue
            if "/case_TR" not in f"/{path}" and "/case_VA" not in f"/{path}" and "/case_TE" not in f"/{path}":
                continue
            shown = _run(["git", "show", f"{ref}:{path}"])
            if shown.returncode != 0 or not shown.stdout:
                continue
            incoming = _load(shown.stdout.encode("utf-8"))
            if incoming is None:
                continue
            dest = REPO / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_file():
                existing = _load(dest.read_bytes())
                if existing is not None and _payload_rank(existing) >= _payload_rank(incoming):
                    skipped.append(path)
                    continue
            dest.write_text(json.dumps(incoming, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            copied.append(f"{ref}:{path}")
    print(json.dumps({"copied": copied, "kept_local": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
