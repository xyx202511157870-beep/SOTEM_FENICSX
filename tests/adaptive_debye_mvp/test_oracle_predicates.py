import numpy as np

from atem3d.adaptive_debye_mvp.oracle_gap import CaseKChoice, evaluate_l0, log_hausdorff_pole_distance


def _choice(case_id, K, ratio, e_b2=0.02, qualifies_b2=False, qualifies_or=True):
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
        qualifies_b2=qualifies_b2,
        qualifies_or=qualifies_or,
    )


def test_log_hausdorff_identical_poles_is_zero():
    tau = np.array([1.0e-4, 1.0e-3, 1.0e-2])
    assert log_hausdorff_pole_distance(tau, tau) == 0.0
    assert log_hausdorff_pole_distance(tau, tau * 10.0) == 1.0


def test_l0_fails_when_ratio_near_one():
    results = []
    for index in range(8):
        choices = [_choice(f"PG{index+1:02d}", k, 0.98, qualifies_b2=True, qualifies_or=True) for k in (4, 6, 8, 10, 12)]
        results.append({"choices": choices, "tasks": [], "official_variant": "S0"})
    l0 = evaluate_l0(results)
    assert l0["passed"] is False
    assert l0["passed_A"] is False
    assert l0["status"] == "L0_FAIL"


def test_l0_ratio_ci_is_on_the_ratio_not_a_shifted_difference():
    results = []
    for index in range(8):
        choices = [_choice(f"PG{index+1:02d}", k, 0.64) for k in (4, 6, 8, 10, 12)]
        results.append({"choices": choices, "tasks": [], "official_variant": "S0"})
    l0 = evaluate_l0(results)
    row = l0["same_k"]["10"]
    assert row["median_ratio"] == 0.64
    assert row["bootstrap_ci_low"] == 0.64
    assert row["bootstrap_ci_high"] == 0.64
    assert row["bootstrap_ci_high"] != 0.0
