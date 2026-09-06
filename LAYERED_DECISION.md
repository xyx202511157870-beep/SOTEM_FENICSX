# LAYERED_DECISION

## 1. FINAL STATUS

L0 passed. L1 is frozen. L2 passed (`L2_PASS`; layered outcome `3D_AUTHORIZED_PENDING_PREFLIGHT`). Flow 5 G3D-0 preflight: BLOCKED. FINAL STATUS=`BLOCKED_BY_SOFTWARE_OR_RESOURCES`. 3-D was not started.

## 2. 3-D continue?

NO for now. L2 authorized 3-D pending preflight. Flow 5 G3D-0 is BLOCKED: this VM has no real FEniCSx; D3-A M1 K=20 needs ≥64 GB (this VM has 16 GB); a fair REF+B2+P-R campaign is 17.7–41 h core / 56–133 h full versus the 10 h cap. Not `need xin decision`. Not `GO_3D_TESTS`.

## 3. Numbers

- L0 pass/fail: `L0_PASS` (A=`True`, B=`False`)
- L0 best same-K OR/B2 median ratio: `0.6830103398631026` at K=`10`
- L0 bootstrap 95% CI: `[0.29918706812649926, 0.9919163382367966]`
- L0 win rate: `1.0`
- L1 frozen: `yes`
- L2 pass/fail: `L2_PASS` (A=`True`, B=`False`)
- same-K P-R/B2 median ratio: `0.382566169465141` at K=`10`
- bootstrap 95% CI: `[0.163401575828889, 0.7770330921368851]`
- win rate: `0.9`
- qualifying-K difference: median `0.0`, nonnegative rate `1.0`
- group outcomes: `{'waveforms': True, 'receivers': True, 'components': True, 'ip': True}`
- 3-D authorized: `yes_pending_preflight`
- n independent-test cases: `10`
- G3D-0: `BLOCKED`
- official 3-D numerical floor: `1.805491133e-4` (= min(0.3%, ⅓ of official L2 median B2−P-R gap `5.416473400e-4`))
- official L2 median `e_pr`: `4.0380472011408475e-4`
- official L2 median `e_b2`: `1.0221694364605988e-3`
- REF reuse: `no`
- Flow 5 decision: `BLOCKED_BY_SOFTWARE_OR_RESOURCES`

## 4. Git / tests

Official L0/L2 call `compute_layered_response`. Flow 5 wrote `generated/receiver_adaptive_debye_mvp/flow5_3d_preflight/` (budget, case contract, reference plan, G3D-0 status) only. No solver-core edits. PR 10 remains draft. No Flow 6. No 3-D forwards.

## 5. If stopped

`BLOCKED_BY_SOFTWARE_OR_RESOURCES`

Explicitly not run: any 3-D FEniCSx forward, `paper_algorithm/run_ip_debye_sweep.py`, `paper_algorithm/run_algorithm_paper.sh` preflight, mesh generation, or G3D-1.
