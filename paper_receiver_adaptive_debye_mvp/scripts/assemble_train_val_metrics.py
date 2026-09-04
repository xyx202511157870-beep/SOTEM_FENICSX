#!/usr/bin/env python3
"""Assemble point-only train/val ranking CSVs. Does not freeze templates."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.oracle_gap import assemble_train_val_metrics


def main() -> int:
    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    payload = assemble_train_val_metrics(
        generated / "flow3_selector",
        generated_dir=generated,
    )
    print(payload["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
