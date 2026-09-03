"""Leakage and authorization guards for the ROADS layered pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io import read_json
from .protocol_constants import TEST_ONLY_SPLITS, TRAIN_ALLOWED_SPLITS


class IndependentTestLeakageError(RuntimeError):
    """Raised when a train/validation stage reads independent-test artifacts."""


class ThreeDNotAuthorizedError(RuntimeError):
    """Raised when a 3-D path is requested before L2 authorization."""


def assert_split_readable(split: str, *, stage: str) -> None:
    """Forbid train/selector stages from reading independent-test splits."""

    name = str(split)
    stage_name = str(stage)
    if stage_name in {"train", "validation", "selector", "flow3"} and name in TEST_ONLY_SPLITS:
        raise IndependentTestLeakageError(
            f"stage {stage_name!r} is forbidden from reading split {name!r}"
        )
    if stage_name in {"train", "selector", "flow3"} and name not in TRAIN_ALLOWED_SPLITS:
        raise IndependentTestLeakageError(
            f"stage {stage_name!r} cannot read unregistered split {name!r}"
        )


def assert_records_split_safe(records: Iterable[dict], *, stage: str, split_key: str = "split") -> None:
    """Reject a table that contains independent-test rows during training."""

    for record in records:
        if split_key in record:
            assert_split_readable(str(record[split_key]), stage=stage)


def layered_gate_path(root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path("generated") / "receiver_adaptive_debye_mvp"
    return Path(base) / "LAYERED_GATE_PASSED.json"


def assert_3d_authorized(root: str | Path | None = None) -> dict:
    """Require a frozen L2 authorization file before any 3-D work."""

    path = layered_gate_path(root)
    if not path.is_file():
        raise ThreeDNotAuthorizedError(
            "3-D is forbidden until L2 writes LAYERED_GATE_PASSED.json; "
            f"missing {path}"
        )
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ThreeDNotAuthorizedError("LAYERED_GATE_PASSED.json must be a mapping")
    status = str(payload.get("status", ""))
    if status != "3D_AUTHORIZED_PENDING_PREFLIGHT":
        raise ThreeDNotAuthorizedError(
            "3-D is forbidden because layered status is "
            f"{status!r}, not 3D_AUTHORIZED_PENDING_PREFLIGHT"
        )
    if not bool(payload.get("l2_passed", False)):
        raise ThreeDNotAuthorizedError("3-D is forbidden because l2_passed is not true")
    return payload


def refuse_3d_before_l2(root: str | Path | None = None) -> None:
    """Always-on 3-D guard used by later-stage scripts."""

    assert_3d_authorized(root)
