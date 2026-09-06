#!/usr/bin/env python3
"""Flow 1: freeze protocol, cases, and candidate templates."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.io import sha256_hex, write_json
from atem3d.adaptive_debye_mvp.registry import freeze_protocol_artifacts


def main() -> int:
    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    paper = REPO_ROOT / "paper_receiver_adaptive_debye_mvp"
    protocol = paper / "protocol.md"
    if not protocol.is_file():
        raise SystemExit("protocol.md is missing")
    payload = freeze_protocol_artifacts(generated)
    payload["protocol_md_sha256"] = sha256_hex(protocol.read_text(encoding="utf-8"))
    write_json(generated / "protocol_hash.json", payload)
    digest = hashlib.sha256()
    for name in ("case_registry.csv", "candidate_registry.csv", "registry_manifest.json", "protocol_hash.json"):
        digest.update(Path(generated / name).read_bytes())
    write_json(
        generated / "FLOW1_STATUS.json",
        {"status": "FROZEN", "bundle_sha256": digest.hexdigest(), **payload},
    )
    print(f"froze {payload['n_cases']} cases and {payload['n_candidates']} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
