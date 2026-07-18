"""Fail-closed validation gates for the two-level SOTEM benchmark."""

from __future__ import annotations

from collections.abc import Mapping
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
        "reference_independent": state
        in {"noip_internally_validated", "ip_internally_validated"},
        "noip_reference_independent": noip_reference_independent,
        "ip_reference_independent": ip_reference_independent,
        "gates": gates_copy,
        "gate_results": gate_results,
        "failed_gates": failed_gates,
        "reason_codes": reason_codes,
    }
