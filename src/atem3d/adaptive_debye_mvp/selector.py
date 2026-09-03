"""Train/validation template selector. Independent-test is unreadable here."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .guards import IndependentTestLeakageError, assert_records_split_safe, assert_split_readable
from .io import read_json, write_json, write_records_csv
from .oracle_gap import FitRecord, pick_spectral_best, reduce_case_error, reduce_ip_error, spectral_error
from .protocol_constants import K_PILOT, TEST_ONLY_SPLITS


def _require_l0_pass(l0_path: str | Path) -> dict:
    payload = read_json(l0_path)
    if not isinstance(payload, dict) or not bool(payload.get("passed", False)):
        raise RuntimeError("selector cannot run unless L0 passed")
    return payload


def _rank_templates(scores: dict[str, list[float]], spectral: dict[str, float], conditions: dict[str, float]) -> list[str]:
    def key(template_id: str) -> tuple:
        values = scores[template_id]
        fail = float(sum(value >= 1.0e300 for value in values) / max(len(values), 1))
        p95 = float(sorted(values)[int(0.95 * (len(values) - 1))]) if values else float("inf")
        return (fail, p95, spectral.get(template_id, float("inf")), conditions.get(template_id, float("inf")), template_id)

    return sorted(scores, key=key)


def select_templates(
    *,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    l0_path: str | Path,
    output_dir: str | Path,
    k_values: tuple[int, ...] = K_PILOT,
) -> dict[str, Any]:
    """Lexicographic train ranking, then one validation pick per K.

    ``train_records`` / ``validation_records`` must not contain independent_test.
    Each record needs ``split``, ``K``, ``candidate_id``, ``total_p95``,
    ``spectral_error`` and ``condition_number``.
    """

    _require_l0_pass(l0_path)
    assert_records_split_safe(train_records, stage="selector")
    assert_records_split_safe(validation_records, stage="selector")
    for record in [*train_records, *validation_records]:
        if str(record.get("split")) in TEST_ONLY_SPLITS:
            raise IndependentTestLeakageError("selector received an independent-test or pressure record")

    selected: dict[str, str] = {}
    spectral_selected: dict[str, str] = {}
    audit: dict[str, Any] = {"k_values": list(k_values), "train_top": {}, "validation_pick": {}}
    train_rows = []
    val_rows = []
    for poles in k_values:
        train_k = [row for row in train_records if int(row["K"]) == poles]
        val_k = [row for row in validation_records if int(row["K"]) == poles]
        scores: dict[str, list[float]] = defaultdict(list)
        spectral: dict[str, float] = {}
        conditions: dict[str, float] = {}
        spec_scores: dict[str, list[float]] = defaultdict(list)
        for row in train_k:
            template = str(row["candidate_id"])
            scores[template].append(float(row["total_p95"]))
            spec_scores[template].append(float(row["spectral_error"]))
            spectral[template] = float(row["spectral_error"])
            conditions[template] = float(row["condition_number"])
            train_rows.append(row)
        ranking = _rank_templates(scores, spectral, conditions)
        spectral_ranking = _rank_templates(spec_scores, spectral, conditions)
        top = ranking[: max(3, min(5, len(ranking)))]
        if not top:
            continue
        best_val = None
        best_score = float("inf")
        for template in top:
            values = [float(row["total_p95"]) for row in val_k if str(row["candidate_id"]) == template]
            score = float(sorted(values)[int(0.95 * (len(values) - 1))]) if values else float("inf")
            if score < best_score:
                best_score = score
                best_val = template
        selected[str(poles)] = best_val or top[0]
        spectral_selected[str(poles)] = spectral_ranking[0]
        audit["train_top"][str(poles)] = top
        audit["validation_pick"][str(poles)] = selected[str(poles)]
        val_rows.extend(val_k)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "selected_template_by_K.json", selected)
    write_json(root / "spectral_selected_template_by_K.json", spectral_selected)
    write_records_csv(root / "train_candidate_metrics.csv", train_rows)
    write_records_csv(root / "validation_candidate_metrics.csv", val_rows)
    write_json(
        root / "selector_audit.json",
        {
            **audit,
            "used_receiver_data_for_b2": False,
            "independent_test_unread": True,
            "l0_required": True,
        },
    )
    return {"selected": selected, "spectral_selected": spectral_selected, "audit": audit}


def load_independent_test_for_selector(records: list[dict[str, Any]]) -> None:
    """Explicit leakage trap used by tests and later stages."""

    assert_records_split_safe(records, stage="selector")
    raise IndependentTestLeakageError("selector must not load independent_test")
