#!/usr/bin/env python3
"""Flow 3: deployable selector. Refuses to run unless L0 passed."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atem3d.adaptive_debye_mvp.guards import IndependentTestLeakageError, assert_split_readable
from atem3d.adaptive_debye_mvp.io import read_csv, read_json, write_json
from atem3d.adaptive_debye_mvp.selector import select_templates


def main() -> int:
    generated = REPO_ROOT / "generated" / "receiver_adaptive_debye_mvp"
    l0_path = generated / "flow2_oracle_gap" / "L0_summary.json"
    if not l0_path.is_file():
        raise SystemExit("L0_summary.json is missing; Flow 3 is unauthorized")
    l0 = read_json(l0_path)
    if not bool(l0.get("passed", False)):
        write_json(
            generated / "FLOW3_STATUS.json",
            {"status": "REFUSED", "reason": "L0 did not pass", "three_d_run": False},
        )
        raise SystemExit("Flow 3 refused: L0 did not pass")

    train_path = generated / "flow3_selector" / "incoming_train.csv"
    val_path = generated / "flow3_selector" / "incoming_validation.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise SystemExit("selector inputs are missing; generate train/val metrics first")
    train = read_csv(train_path)
    validation = read_csv(val_path)
    for row in [*train, *validation]:
        assert_split_readable(str(row["split"]), stage="selector")
        if str(row["split"]) == "independent_test":
            raise IndependentTestLeakageError("train_selector.py read independent_test")
    payload = select_templates(
        train_records=train,
        validation_records=validation,
        l0_path=l0_path,
        output_dir=generated / "flow3_selector",
    )
    write_json(generated / "FLOW3_STATUS.json", {"status": "L1_FROZEN", "selected": payload["selected"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
