# LAYERED_DECISION

## 1. FINAL STATUS

L0 passed. L1 is frozen. L2 passed (`L2_PASS`). FINAL STATUS=`3D_AUTHORIZED_PENDING_PREFLIGHT`. 3-D was not started.

## 2. 3-D continue?

YES (authorized pending preflight). 3-D was not run.

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

## 4. Git / tests

Official L0/L2 call `compute_layered_response`. PR 10 remains draft. No 3-D forwards.

## 5. If stopped

`3D_AUTHORIZED_PENDING_PREFLIGHT`

Explicitly not run: any 3-D FEniCSx forward, including `paper_algorithm/run_ip_debye_sweep.py`.
