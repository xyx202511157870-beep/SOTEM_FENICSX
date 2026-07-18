"""Fail-closed validation gates for the two-level SOTEM benchmark."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any


LEI = ("lei_simpeg", "lei_fenicsx")
SONG_NOIP = ("song_noip_simpeg", "song_noip_fenicsx")
SONG_IP = (
    "song_ip_simpeg",
    "song_ip_fenicsx",
    "song_delta_simpeg",
    "song_delta_fenicsx",
    "material_gate",
)

_REQUIRED_GATES = LEI + SONG_NOIP + SONG_IP
_PLUMBING_REFERENCE = "empymod_plus_zero_secondary"
_NOIP_REFERENCES = ("empymod_noip_layered", "empymod_exact_layered")
_EXACT_IP_REFERENCE = "empymod_exact_layered"


def _strict_true(value: object) -> bool:
    return type(value) is bool and value is True


def _is_provenance(value: object, accepted: tuple[str, ...]) -> bool:
    return type(value) is str and value in accepted


def _effective_provenance(
    gates: Mapping[str, Any], split_name: str
) -> object:
    if split_name in gates:
        return gates[split_name]
    return gates.get("reference_provenance")


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _snapshot_mapping(
    value: Mapping[Any, Any], active_ids: set[int]
) -> dict[str, Any]:
    entries: list[tuple[str, bool, str, Any]] = []
    for key, item in value.items():
        string_key = type(key) is str
        if string_key:
            snapshot_key = key
        else:
            key_snapshot = _snapshot_value(key, active_ids)
            snapshot_key = (
                f"__key__:{_qualified_type_name(key)}:"
                f"{_canonical_json(key_snapshot)}"
            )
        item_snapshot = _snapshot_value(item, active_ids)
        entries.append(
            (
                snapshot_key,
                string_key,
                _canonical_json(item_snapshot),
                item_snapshot,
            )
        )

    result: dict[str, Any] = {}
    reserved_string_keys = {entry[0] for entry in entries if entry[1]}
    for snapshot_key, string_key, _, item_snapshot in sorted(
        entries, key=lambda entry: (entry[0], not entry[1], entry[2])
    ):
        if string_key:
            result[snapshot_key] = item_snapshot
            continue
        unique_key = snapshot_key
        suffix = 2
        while unique_key in result or unique_key in reserved_string_keys:
            unique_key = f"{snapshot_key}#{suffix}"
            suffix += 1
        result[unique_key] = item_snapshot
    return result


def _snapshot_value(value: Any, active_ids: set[int]) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"__float__": "nan"}
        return {"__float__": "+inf" if value > 0 else "-inf"}

    is_container = isinstance(value, (Mapping, list, tuple, set, frozenset))
    if is_container and id(value) in active_ids:
        return {"__cycle__": True}
    if is_container:
        active_ids.add(id(value))
        try:
            if isinstance(value, Mapping):
                return _snapshot_mapping(value, active_ids)
            if isinstance(value, (list, tuple)):
                return [_snapshot_value(item, active_ids) for item in value]
            snapshots = [_snapshot_value(item, active_ids) for item in value]
            return sorted(snapshots, key=_canonical_json)
        finally:
            active_ids.remove(id(value))

    return {"__type__": _qualified_type_name(value)}


def _snapshot_evidence(value: Mapping[Any, Any]) -> dict[str, Any]:
    return _snapshot_value(value, set())


def summarize_sotem_gates(gates: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize the fail-closed noIP and IP validation state.

    Required gate values pass only when they are the literal Boolean ``True``.
    Provenance fields are evaluated independently for the noIP and IP levels.
    """
    if not isinstance(gates, Mapping):
        raise ValueError("SOTEM gates input must be a mapping")

    gates_copy = dict(gates)
    gate_results = {
        name: _strict_true(gates_copy.get(name)) for name in _REQUIRED_GATES
    }
    failed_gates = [name for name in _REQUIRED_GATES if not gate_results[name]]

    noip_provenance = _effective_provenance(
        gates_copy, "noip_reference_provenance"
    )
    ip_provenance = _effective_provenance(
        gates_copy, "ip_reference_provenance"
    )
    noip_reference_independent = _is_provenance(
        noip_provenance, _NOIP_REFERENCES
    )
    ip_reference_independent = _is_provenance(
        ip_provenance, (_EXACT_IP_REFERENCE,)
    )
    plumbing_reference = (
        _is_provenance(noip_provenance, (_PLUMBING_REFERENCE,))
        and _is_provenance(ip_provenance, (_PLUMBING_REFERENCE,))
    )

    noip_gates_pass = all(gate_results[name] for name in LEI + SONG_NOIP)
    ip_gates_pass = all(gate_results[name] for name in SONG_IP)
    all_gates_pass = not failed_gates
    noip_internally_validated = (
        noip_gates_pass and noip_reference_independent
    )
    ip_internally_validated = (
        noip_internally_validated
        and ip_gates_pass
        and ip_reference_independent
    )

    if ip_internally_validated:
        state = "ip_internally_validated"
    elif noip_internally_validated:
        state = "noip_internally_validated"
    elif all_gates_pass and plumbing_reference:
        state = "plumbing_pass"
    else:
        state = "failed_with_reproducible_evidence"

    reason_codes = [
        f"missing_or_failed_gate:{name}" for name in failed_gates
    ]
    any_circular_reference = (
        _is_provenance(noip_provenance, (_PLUMBING_REFERENCE,))
        or _is_provenance(ip_provenance, (_PLUMBING_REFERENCE,))
    )
    if any_circular_reference:
        reason_codes.append("circular_reference")
    if plumbing_reference:
        reason_codes.append("plumbing_only_reference")
    else:
        if not noip_reference_independent:
            reason_codes.append("noip_reference_not_independent")
        if not ip_reference_independent:
            reason_codes.append("ip_reference_not_exact")

    return {
        "state": state,
        "noip_internally_validated": noip_internally_validated,
        "ip_internally_validated": ip_internally_validated,
        "reference_independent": noip_reference_independent
        or ip_reference_independent,
        "noip_reference_independent": noip_reference_independent,
        "ip_reference_independent": ip_reference_independent,
        "gates": _snapshot_evidence(gates_copy),
        "gate_results": gate_results,
        "failed_gates": failed_gates,
        "reason_codes": reason_codes,
    }
