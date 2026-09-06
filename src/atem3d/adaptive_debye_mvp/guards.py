"""Leakage and authorization guards for the ROADS layered pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io import read_json, sha256_file
from .protocol_constants import SPLIT_PREFIX, TEST_ONLY_SPLITS, TRAIN_ALLOWED_SPLITS


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


def assert_case_ids_split_safe(case_ids: Iterable[str], *, stage: str) -> None:
    """Reject independent-test / pressure prefixes during train/selector stages."""

    forbidden = []
    if stage in {"train", "validation", "selector", "flow3"}:
        forbidden.extend([SPLIT_PREFIX["independent_test"], SPLIT_PREFIX["layered_pressure"]])
    if stage in {"selector", "flow3"}:
        forbidden.append(SPLIT_PREFIX["pilot_gap"])
    for case_id in case_ids:
        prefix = "".join(ch for ch in str(case_id) if ch.isalpha())
        if prefix in forbidden:
            raise IndependentTestLeakageError(
                f"stage {stage!r} cannot read case id {case_id!r}"
            )


def assert_file_hash(path: str | Path, expected_sha256: str) -> str:
    digest = sha256_file(path)
    if digest != str(expected_sha256):
        raise ValueError(f"hash mismatch for {path}: {digest} != {expected_sha256}")
    return digest


def snapshot_cache_keys(cache_dir: str | Path) -> frozenset[str]:
    root = Path(cache_dir)
    if not root.is_dir():
        return frozenset()
    return frozenset(path.name for path in root.glob("*.npz"))


def assert_cache_untouched(
    cache_dir: str | Path,
    *,
    before: Iterable[str],
    forbidden_prefixes: tuple[str, ...] = ("TE", "LP", "PG"),
    forbidden_tokens: tuple[str, ...] = (":W3:", ":W4:"),
) -> list[str]:
    """Return new cache keys and reject leakage prefixes/tokens."""

    after = snapshot_cache_keys(cache_dir)
    new_keys = sorted(after - frozenset(before))
    for name in new_keys:
        if any(name.startswith(prefix) for prefix in forbidden_prefixes):
            raise IndependentTestLeakageError(f"cache wrote forbidden prefix key {name}")
        stem = Path(name).stem
        waveform = stem.rsplit(":", 1)[-1]
        if any(token.strip(":") == waveform or token in stem for token in forbidden_tokens):
            raise IndependentTestLeakageError(f"cache wrote forbidden token key {name}")
    return new_keys
