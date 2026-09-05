# LAYERED_DECISION

## 1. FINAL STATUS

L0 passed. L1 is frozen (`FLOW3_STATUS=L1_FROZEN`, official variant S1, train/val only). Official point+disk JSON exist for TR01–TR12 and VA01–VA06. L2 on independent_test is next. 3-D was not run.

## 2. 3-D continue?

NO. L1 is frozen but L2 has not passed. 3-D was not run.

## 3. Numbers

- L0 pass/fail: `L0_PASS` (A=`True`, B=`False`)
- best same-K OR/B2 median ratio: `0.6830103398631026` at K=`10`
- bootstrap 95% CI: `[0.29918706812649926, 0.9919163382367966]`
- win rate: `1.0`
- qualifying-K oracle difference: median `0.0`, nonnegative rate `1.0`
- n official pilot cases with disks: `8`
- L1 frozen: `yes` (`FLOW3_STATUS=L1_FROZEN`; 18/18 official disk JSON; independent_test unread)
- L1 official variant: `S1`
- L1 P-R: K4=`K04_cc_span4.0_shift-0.5_dens1.25`, K6=`K06_cc_span4.0_shift+0.0_dens1.25`, K8=`K08_cc_span4.0_shift-0.5_dens1.00`, K10=`K10_cc_span4.0_shift-0.5_dens1.00`, K12=`K12_cc_span6.0_shift+0.0_dens1.25`
- L1 B2 spectral: differs from P-R at K=6, 10, 12
- L2 pass/fail: not assembled yet
- same-K P-R/B2 median ratio: n/a
- bootstrap CI: n/a
- win rate: n/a
- qualifying-K difference: n/a
- group outcomes: n/a
- 3-D authorized: `no`

## 4. Git / tests

Official L0 calls `compute_layered_response`. Flow 3/4 reuse `evaluate_pilot_case`. L2 compares frozen P-R to frozen spectral B2 (no test reselection). PR 10 remains draft.

## 5. If stopped

Not stopped at L0. L1 frozen. L2 not finished. 3-D still unauthorized.

Explicitly not run: any 3-D FEniCSx forward, including `paper_algorithm/run_ip_debye_sweep.py`.
