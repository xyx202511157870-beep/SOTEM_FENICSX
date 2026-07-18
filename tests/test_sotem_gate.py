from copy import deepcopy

import pytest

from atem3d.sotem_gate import summarize_sotem_gates


GATE_NAMES = (
    "lei_simpeg",
    "lei_fenicsx",
    "song_noip_simpeg",
    "song_noip_fenicsx",
    "song_ip_simpeg",
    "song_ip_fenicsx",
    "song_delta_simpeg",
    "song_delta_fenicsx",
    "material_gate",
)


def passing_gates(provenance="empymod_exact_layered"):
    gates = {name: True for name in GATE_NAMES}
    gates["reference_provenance"] = provenance
    return gates


def test_empymod_plus_zero_secondary_is_plumbing_only():
    summary = summarize_sotem_gates(
        {
            "lei_simpeg": True,
            "lei_fenicsx": True,
            "song_noip_simpeg": True,
            "song_noip_fenicsx": True,
            "song_ip_simpeg": True,
            "song_ip_fenicsx": True,
            "song_delta_simpeg": True,
            "song_delta_fenicsx": True,
            "material_gate": True,
            "reference_provenance": "empymod_plus_zero_secondary",
        }
    )

    assert summary["state"] == "plumbing_pass"
    assert summary["ip_internally_validated"] is False


def test_all_independent_gates_produce_ip_internal_validation():
    gates = {name: True for name in GATE_NAMES}
    gates["reference_provenance"] = "empymod_exact_layered"

    assert summarize_sotem_gates(gates)["state"] == "ip_internally_validated"


def test_failed_ip_gate_preserves_noip_internal_validation():
    gates = passing_gates()
    gates["material_gate"] = False

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "noip_internally_validated"
    assert summary["noip_internally_validated"] is True
    assert summary["ip_internally_validated"] is False
    assert summary["reference_independent"] is True
    assert summary["failed_gates"] == ["material_gate"]
    assert summary["reason_codes"] == ["missing_or_failed_gate:material_gate"]


def test_noip_reference_cannot_validate_ip_when_all_gates_pass():
    summary = summarize_sotem_gates(passing_gates("empymod_noip_layered"))

    assert summary["state"] == "noip_internally_validated"
    assert summary["noip_reference_independent"] is True
    assert summary["ip_reference_independent"] is False
    assert summary["reason_codes"] == ["ip_reference_not_exact"]


def test_circular_reference_with_all_gates_passes_plumbing_only():
    summary = summarize_sotem_gates(
        passing_gates("empymod_plus_zero_secondary")
    )

    assert summary["state"] == "plumbing_pass"
    assert summary["noip_internally_validated"] is False
    assert summary["ip_internally_validated"] is False
    assert summary["reference_independent"] is False
    assert summary["noip_reference_independent"] is False
    assert summary["ip_reference_independent"] is False
    assert summary["reason_codes"] == [
        "circular_reference",
        "plumbing_only_reference",
    ]


def test_circular_reference_with_failed_gate_fails_closed():
    gates = passing_gates("empymod_plus_zero_secondary")
    gates["song_delta_simpeg"] = False

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["failed_gates"] == ["song_delta_simpeg"]
    assert summary["reason_codes"] == [
        "missing_or_failed_gate:song_delta_simpeg",
        "circular_reference",
        "plumbing_only_reference",
    ]


@pytest.mark.parametrize("provenance", [None, "", "unreviewed_reference"])
def test_unknown_or_empty_provenance_fails_closed(provenance):
    gates = passing_gates(provenance)

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["reason_codes"] == [
        "noip_reference_not_independent",
        "ip_reference_not_exact",
    ]


@pytest.mark.parametrize("provenance", [[], {"source": "empymod"}])
def test_structured_unknown_provenance_fails_closed_without_raising(provenance):
    gates = passing_gates(provenance)

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["reason_codes"] == [
        "noip_reference_not_independent",
        "ip_reference_not_exact",
    ]


def test_missing_provenance_fails_closed():
    gates = passing_gates()
    del gates["reference_provenance"]

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["reason_codes"] == [
        "noip_reference_not_independent",
        "ip_reference_not_exact",
    ]


def test_missing_gates_fail_with_deterministic_ordered_reasons():
    gates = {
        "lei_fenicsx": True,
        "song_noip_fenicsx": True,
        "reference_provenance": "empymod_exact_layered",
    }

    summary = summarize_sotem_gates(gates)

    expected_failed = [
        "lei_simpeg",
        "song_noip_simpeg",
        "song_ip_simpeg",
        "song_ip_fenicsx",
        "song_delta_simpeg",
        "song_delta_fenicsx",
        "material_gate",
    ]
    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["failed_gates"] == expected_failed
    assert summary["reason_codes"] == [
        f"missing_or_failed_gate:{name}" for name in expected_failed
    ]
    assert list(summary["gate_results"]) == list(GATE_NAMES)


def test_only_literal_boolean_true_passes_and_input_is_not_mutated():
    gates = passing_gates()
    gates["lei_simpeg"] = "true"
    gates["lei_fenicsx"] = 1
    gates["song_noip_simpeg"] = {"passed": True}
    original = deepcopy(gates)

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["failed_gates"][:3] == [
        "lei_simpeg",
        "lei_fenicsx",
        "song_noip_simpeg",
    ]
    assert summary["gate_results"]["lei_simpeg"] is False
    assert summary["gate_results"]["lei_fenicsx"] is False
    assert summary["gate_results"]["song_noip_simpeg"] is False
    assert summary["gates"] == original
    assert summary["gates"] is not gates
    assert gates == original


def test_split_provenance_fields_override_legacy_for_each_level():
    gates = passing_gates("empymod_plus_zero_secondary")
    gates["noip_reference_provenance"] = "empymod_noip_layered"
    gates["ip_reference_provenance"] = "empymod_exact_layered"

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "ip_internally_validated"
    assert summary["noip_reference_independent"] is True
    assert summary["ip_reference_independent"] is True
    assert summary["reason_codes"] == []


def test_split_ip_provenance_override_prevents_legacy_exact_ip_validation():
    gates = passing_gates("empymod_exact_layered")
    gates["ip_reference_provenance"] = "empymod_noip_layered"

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "noip_internally_validated"
    assert summary["reason_codes"] == ["ip_reference_not_exact"]


def test_split_noip_provenance_override_fails_before_ip_level():
    gates = passing_gates("empymod_exact_layered")
    gates["noip_reference_provenance"] = ""

    summary = summarize_sotem_gates(gates)

    assert summary["state"] == "failed_with_reproducible_evidence"
    assert summary["noip_reference_independent"] is False
    assert summary["ip_reference_independent"] is True
    assert summary["reason_codes"] == ["noip_reference_not_independent"]


def test_split_fields_fall_back_independently_and_accept_explicit_exact_noip():
    noip_override = passing_gates("empymod_exact_layered")
    noip_override["noip_reference_provenance"] = "empymod_noip_layered"
    explicit_exact_noip = passing_gates("empymod_noip_layered")
    explicit_exact_noip["noip_reference_provenance"] = "empymod_exact_layered"
    explicit_exact_noip["ip_reference_provenance"] = "empymod_exact_layered"

    assert summarize_sotem_gates(noip_override)["state"] == (
        "ip_internally_validated"
    )
    assert summarize_sotem_gates(explicit_exact_noip)["state"] == (
        "ip_internally_validated"
    )


@pytest.mark.parametrize("value", [None, [], "gates", 1])
def test_non_mapping_input_is_rejected(value):
    with pytest.raises(ValueError, match="mapping"):
        summarize_sotem_gates(value)


def test_repeated_calls_return_equal_but_independent_results():
    gates = passing_gates()

    first = summarize_sotem_gates(gates)
    second = summarize_sotem_gates(gates)

    assert first == second
    assert first is not second
    assert first["gates"] is not second["gates"]
    assert first["gate_results"] is not second["gate_results"]
