"""Train/validation template selector. Independent-test is unreadable here."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .guards import IndependentTestLeakageError, assert_records_split_safe, assert_split_readable
from .io import read_json, write_json, write_records_csv
from .oracle_gap import (
    hydrate_result,
    reduce_case_error,
    reduce_ip_error,
    tasks_qualify,
)
from .protocol_constants import K_PILOT, TEST_ONLY_SPLITS


KEY_DEFINITION = (
    "fail_case_rate, Q0.95(total_p95), Q0.95(ip_p95), worst_group_Q0.95, "
    "median(total_p95), median(spectral_error), median(condition_number), candidate_id"
)
GROUP_FIELDS = (
    "group_p95_W0",
    "group_p95_W1",
    "group_p95_W2",
    "group_p95_W3",
    "group_p95_point",
    "group_p95_disk_1.0",
    "group_p95_disk_4.0",
    "h_p95",
    "dbdt_p95",
)


def _require_l0_pass(l0_path: str | Path) -> dict:
    payload = read_json(l0_path)
    if not isinstance(payload, dict) or not bool(payload.get("passed", False)):
        raise RuntimeError("selector cannot run unless L0 passed")
    return payload


def _finite(values) -> list[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _q95(values) -> float:
    finite = _finite(values)
    if not finite:
        return float("inf")
    return float(np.quantile(finite, 0.95))


def _median(values) -> float:
    finite = _finite(values)
    return float(np.median(finite)) if finite else float("inf")


def template_lex_key(rows: list[dict[str, Any]]) -> tuple:
    """Protocol lexicographic key (lower is better) over case-level rows."""

    if not rows:
        return (1.0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), "")
    fails = []
    for row in rows:
        if "qualifies" in row:
            fails.append(0.0 if bool(row["qualifies"]) else 1.0)
        else:
            fails.append(1.0 if float(row.get("total_p95", float("inf"))) >= 1.0e300 else 0.0)
    group_q95 = []
    for field in GROUP_FIELDS:
        values = [row[field] for row in rows if field in row and np.isfinite(float(row[field]))]
        if values:
            group_q95.append(_q95(values))
    template_id = str(rows[0]["candidate_id"])
    return (
        float(np.mean(fails)),
        _q95(row["total_p95"] for row in rows),
        _q95(row.get("ip_p95", row.get("ip_increment_nrmse", float("inf"))) for row in rows),
        max(group_q95) if group_q95 else 0.0,
        _median(row["total_p95"] for row in rows),
        _median(row.get("spectral_error", float("inf")) for row in rows),
        _median(row.get("condition_number", float("inf")) for row in rows),
        template_id,
    )


def _rank_templates(rows_by_template: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(rows_by_template, key=lambda template: template_lex_key(rows_by_template[template]))


def _spectral_rank(rows_by_template: dict[str, list[dict[str, Any]]]) -> list[str]:
    def key(template_id: str) -> tuple:
        rows = rows_by_template[template_id]
        return (
            _median(row.get("spectral_error", float("inf")) for row in rows),
            _median(row.get("condition_number", float("inf")) for row in rows),
            template_id,
        )

    return sorted(rows_by_template, key=key)


def _group_by_template(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_id"])].append(row)
    return grouped


def _scope_of(row: dict[str, Any]) -> str:
    return str(row.get("scope") or "point")


def records_from_case_results(
    results: list[dict[str, Any]],
    *,
    split: str,
    official_variant: str,
    scope: str,
) -> list[dict[str, Any]]:
    """Flatten per-case JSON into selector records. Never accepts test splits."""

    assert_split_readable(split, stage="selector")
    rows: list[dict[str, Any]] = []
    for raw in results:
        result = hydrate_result(raw)
        case_id = result["case_id"]
        if str(case_id).startswith(("TE", "LP", "PG")) and split in {"train", "validation"}:
            raise IndependentTestLeakageError(f"selector record builder saw forbidden case {case_id}")
        allowed = {"point"} if scope == "point" else {"point", "disk_1.0", "disk_4.0"}
        by: dict[tuple[int, str], list] = defaultdict(list)
        for task in result["tasks"]:
            if task.receiver_id not in allowed:
                continue
            by[(int(task.K), str(task.candidate_id))].append(task)
        fits = {item["candidate_id"]: item for item in result.get("fit_summary") or []}
        for (poles, candidate_id), tasks in by.items():
            fit = fits.get(candidate_id, {})
            spectral = fit.get("spectral_error_s1" if official_variant == "S1" else "spectral_error_s0", float("nan"))
            waveform_groups = {}
            for waveform_id in ("W0", "W1", "W2", "W3"):
                subset = [item for item in tasks if item.waveform_id == waveform_id]
                if subset:
                    waveform_groups[f"group_p95_{waveform_id}"] = reduce_case_error(subset)
            receiver_groups = {}
            for receiver_id in ("point", "disk_1.0", "disk_4.0"):
                subset = [item for item in tasks if item.receiver_id == receiver_id]
                if subset:
                    receiver_groups[f"group_p95_{receiver_id}"] = reduce_case_error(subset)
            h_values = [item.h_p95 for item in tasks if np.isfinite(item.h_p95)]
            db_values = [item.dbdt_p95 for item in tasks if np.isfinite(item.dbdt_p95)]
            rows.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "K": poles,
                    "candidate_id": candidate_id,
                    "valid": bool(fit.get("valid", True)),
                    "scope": scope,
                    "n_tasks": len(tasks),
                    "total_p95": reduce_case_error(tasks),
                    "ip_p95": reduce_ip_error(tasks),
                    "qualifies": tasks_qualify(tasks),
                    "h_p95": float(np.max(h_values)) if h_values else float("nan"),
                    "dbdt_p95": float(np.max(db_values)) if db_values else float("nan"),
                    "spectral_error": float(spectral) if np.isfinite(spectral) else float("nan"),
                    "spectral_error_s0": float(fit.get("spectral_error_s0", float("nan"))),
                    "spectral_error_s1": float(fit.get("spectral_error_s1", float("nan"))),
                    "condition_number": float(fit.get("condition_number", float("nan"))),
                    **waveform_groups,
                    **receiver_groups,
                }
            )
    assert_records_split_safe(rows, stage="selector")
    return rows


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
    audit: dict[str, Any] = {
        "k_values": list(k_values),
        "train_top": {},
        "validation_pick": {},
        "key_definition": KEY_DEFINITION,
        "pilot_gap_used": False,
        "R1": {},
        "T_K": {},
        "R2": {},
        "validation_keys": {},
    }
    train_rows = []
    val_rows = []
    for poles in k_values:
        train_k = [row for row in train_records if int(row["K"]) == poles]
        val_k = [row for row in validation_records if int(row["K"]) == poles]
        train_point = [row for row in train_k if _scope_of(row) == "point"]
        train_disk = [row for row in train_k if _scope_of(row) == "point_disk"]
        val_disk = [row for row in val_k if _scope_of(row) == "point_disk"]
        stage1_rows = train_point or train_k
        stage1 = _group_by_template(stage1_rows)
        if not stage1:
            continue
        r1 = _rank_templates(stage1)
        top = r1[: max(3, min(5, len(r1)))]
        spectral_rows = _group_by_template(stage1_rows)
        spectral_ranking = _spectral_rank(spectral_rows)
        stage2_source = train_disk if train_disk else stage1_rows
        stage2 = {template: [row for row in stage2_source if str(row["candidate_id"]) == template] for template in top}
        r2 = _rank_templates({key: value for key, value in stage2.items() if value}) or top
        val_source = val_disk if val_disk else val_k
        best_val = None
        best_key = None
        validation_keys = {}
        for template in r2:
            values = [row for row in val_source if str(row["candidate_id"]) == template]
            key = template_lex_key(values) if values else (1.0, float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), float("inf"), template)
            validation_keys[template] = [float(item) if isinstance(item, (int, float)) else item for item in key[:-1]] + [template]
            if best_key is None or key < best_key:
                best_key = key
                best_val = template
        selected[str(poles)] = best_val or top[0]
        spectral_selected[str(poles)] = spectral_ranking[0]
        audit["train_top"][str(poles)] = top
        audit["validation_pick"][str(poles)] = selected[str(poles)]
        audit["R1"][str(poles)] = r1
        audit["T_K"][str(poles)] = top
        audit["R2"][str(poles)] = r2
        audit["validation_keys"][str(poles)] = validation_keys
        train_rows.extend(train_k)
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
            "n_train_cases": len({row.get("case_id") for row in train_records if row.get("case_id")}),
            "n_validation_cases": len({row.get("case_id") for row in validation_records if row.get("case_id")}),
        },
    )
    return {"selected": selected, "spectral_selected": spectral_selected, "audit": audit}


def load_independent_test_for_selector(records: list[dict[str, Any]]) -> None:
    """Explicit leakage trap used by tests and later stages."""

    assert_records_split_safe(records, stage="selector")
    raise IndependentTestLeakageError("selector must not load independent_test")
