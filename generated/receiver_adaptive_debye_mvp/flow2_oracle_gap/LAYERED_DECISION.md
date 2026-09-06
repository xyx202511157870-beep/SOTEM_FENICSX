# LAYERED_DECISION

## 1. FINAL STATUS

L0 passed. Selector / independent test were not started in this turn.

## 2. 3-D continue?

NO. L2 has not passed. 3-D was not run.

## 3. Numbers

- L0 pass/fail: `L0_PASS` (A=`True`, B=`False`)
- best same-K OR/B2 median ratio: `0.6830103398631026` at K=`10`
- bootstrap 95% CI: `[0.29918706812649926, 0.9919163382367966]`
- win rate: `1.0`
- qualifying-K oracle difference: median `0.0`, nonnegative rate `1.0`
- n official pilot cases with disks: `8`

## 4. Git / tests

Official L0 calls `compute_layered_response`. PR 10 remains draft.

## 5. If stopped

Not stopped at L0. 3-D still unauthorized.

Explicitly not run: any 3-D FEniCSx forward, including `paper_algorithm/run_ip_debye_sweep.py`.
