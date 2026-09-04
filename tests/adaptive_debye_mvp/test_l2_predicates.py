from atem3d.adaptive_debye_mvp.oracle_gap import CaseKChoice, evaluate_l2


def _choice(case_id, K, ratio, e_b2=0.02):
    return CaseKChoice(
        case_id=case_id,
        K=K,
        spectral_id="b2",
        oracle_id="or",
        ids_differ=True,
        e_b2=e_b2,
        e_or=ratio * e_b2,
        ratio=ratio,
        gap=1.0 - ratio,
        ip_b2=0.02,
        ip_or=0.015,
        log_hausdorff=0.1,
        qualifies_b2=False,
        qualifies_or=True,
    )


def test_l2_ratio_ci_is_on_the_ratio():
    results = []
    pr = {str(k): "pr" for k in (4, 6, 8, 10, 12)}
    for index in range(10):
        choices = [_choice(f"TE{index+1:02d}", k, 0.64) for k in (4, 6, 8, 10, 12)]
        results.append(
            {
                "case_id": f"TE{index+1:02d}",
                "choices": [
                    {
                        **choice.__dict__,
                        "spectral_id": "b2",
                        "oracle_id": "or",
                    }
                    for choice in choices
                ],
                "tasks": [
                    {
                        "case_id": f"TE{index+1:02d}",
                        "candidate_id": cid,
                        "K": k,
                        "waveform_id": "W0",
                        "receiver_id": "point",
                        "receiver_index": 0,
                        "total_p95": (0.64 if cid == "pr" else 1.0) * 0.02,
                        "ip_increment_nrmse": 0.01,
                        "passed": True,
                        "unexplained_sign_flips": 0,
                        "peak_time_error_steps_max": 0.0,
                        "h_p95": 0.01,
                        "dbdt_p95": 0.01,
                    }
                    for k in (4, 6, 8, 10, 12)
                    for cid in ("pr", "b2")
                ],
                "official_variant": "S1",
            }
        )
    # Patch choices so apply_pr_choices finds spectral_id b2 and pr_by_k
    for result in results:
        for item in result["choices"]:
            item["spectral_id"] = "b2"
    l2 = evaluate_l2(results, pr_by_k=pr)
    row = l2["same_k"]["10"]
    assert row["median_ratio"] == 0.64
    assert row["bootstrap_ci_high"] != 0.0
    assert l2["status"] in {"L2_PASS", "L2_FAIL"}
