# LAYERED_DECISION

## 1. FINAL STATUS

L0 passed. L1 selector is in progress (train/validation case forwards). L2 has not been assembled. 3-D was not run.

## 2. 3-D continue?

NO. L1 is not frozen and L2 has not passed. 3-D was not run.

## 3. Numbers

- L0 pass/fail: `L0_PASS` (A=`True`, B=`False`)
- best same-K OR/B2 median ratio: `0.6830103398631026` at K=`10`
- bootstrap 95% CI: `[0.29918706812649926, 0.9919163382367966]`
- win rate: `1.0`
- qualifying-K oracle difference: median `0.0`, nonnegative rate `1.0`
- n official pilot cases with disks: `8`
- L1 frozen: `no` (Flow 3 running; `FLOW3_STATUS` not yet `L1_FROZEN`)
- L2 pass/fail: not assembled
- same-K P-R/B2 median ratio: n/a
- bootstrap CI: n/a
- win rate: n/a
- qualifying-K difference: n/a
- group outcomes: n/a
- 3-D authorized: `no`

## 4. Git / tests

Official L0 calls `compute_layered_response`. Flow 3/4 reuse `evaluate_pilot_case`. L2 compares frozen P-R to frozen spectral B2 (no test reselection). PR 10 remains draft.

## 5. If stopped

Not stopped at L0. L1/L2 not finished. 3-D still unauthorized.

Explicitly not run: any 3-D FEniCSx forward, including `paper_algorithm/run_ip_debye_sweep.py`.
