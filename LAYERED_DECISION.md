# LAYERED_DECISION

This file is the required top-level decision record for the ROADS-Debye-MVP
layered-first empymod falsification. It is updated after Flow 2 (and later
gates if they are authorized).

## 1. FINAL STATUS

Pending Flow 2. The library from `paper/mvp-debye-lib` is merged. 3-D has
not been run.

## 2. 3-D continue?

NO until L2 passes. L2 has not been reached.

## 3. Numbers

Not yet available. Flow 0/1/2 will fill:

- L0 pass/fail
- best same-K OR/B2 median ratio
- bootstrap CI
- win rate
- qualifying-K oracle difference

## 4. Git / tests

See the pull request body after Flow 0 artifacts are written.

## 5. If stopped

Not yet decided. Explicitly not run: any 3-D FEniCSx forward, including
`paper_algorithm/run_ip_debye_sweep.py`.
