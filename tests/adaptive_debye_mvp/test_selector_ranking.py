from atem3d.adaptive_debye_mvp.io import write_json
from atem3d.adaptive_debye_mvp.selector import select_templates, template_lex_key


def _row(case_id, K, candidate_id, total, spectral=0.2, cond=1.0, qualifies=True, scope="point", ip=0.01):
    return {
        "split": "train" if case_id.startswith("TR") else "validation",
        "case_id": case_id,
        "K": K,
        "candidate_id": candidate_id,
        "total_p95": total,
        "ip_p95": ip,
        "spectral_error": spectral,
        "condition_number": cond,
        "qualifies": qualifies,
        "scope": scope,
        "group_p95_W0": total,
        "group_p95_point": total,
    }


def test_lex_key_orders_fail_rate_first():
    good = [_row("TR01", 8, "good", 0.9, qualifies=True)]
    bad = [_row("TR01", 8, "bad", 0.1, qualifies=False)]
    assert template_lex_key(good) < template_lex_key(bad)


def test_select_templates_picks_from_train_top(tmp_path):
    l0 = tmp_path / "L0_summary.json"
    write_json(l0, {"passed": True})
    train = []
    val = []
    for k in (4, 6, 8, 10, 12):
        for index in range(12):
            train.append(_row(f"TR{index+1:02d}", k, "best", 0.2 + 0.01 * index, spectral=0.3))
            train.append(_row(f"TR{index+1:02d}", k, "other", 0.8, spectral=0.05))
        for index in range(6):
            val.append(_row(f"VA{index+1:02d}", k, "best", 0.25, spectral=0.3))
            val.append(_row(f"VA{index+1:02d}", k, "other", 0.9, spectral=0.05))
    payload = select_templates(
        train_records=train,
        validation_records=val,
        l0_path=l0,
        output_dir=tmp_path / "out",
    )
    assert payload["selected"]["10"] == "best"
    assert payload["spectral_selected"]["10"] == "other"


def test_spectral_pick_ignores_receiver_columns(tmp_path):
    l0 = tmp_path / "L0_summary.json"
    write_json(l0, {"passed": True})
    train = []
    val = []
    for k in (4, 6, 8, 10, 12):
        for index in range(12):
            train.append(_row(f"TR{index+1:02d}", k, "spec", 9.0, spectral=0.01))
            train.append(_row(f"TR{index+1:02d}", k, "recv", 0.1, spectral=0.9))
        for index in range(6):
            val.append(_row(f"VA{index+1:02d}", k, "spec", 9.0, spectral=0.01))
            val.append(_row(f"VA{index+1:02d}", k, "recv", 0.1, spectral=0.9))
    first = select_templates(train_records=train, validation_records=val, l0_path=l0, output_dir=tmp_path / "a")
    for row in train:
        row["total_p95"] = 99.0
        row["ip_p95"] = 99.0
    second = select_templates(train_records=train, validation_records=val, l0_path=l0, output_dir=tmp_path / "b")
    assert first["spectral_selected"] == second["spectral_selected"]
    assert first["spectral_selected"]["8"] == "spec"
