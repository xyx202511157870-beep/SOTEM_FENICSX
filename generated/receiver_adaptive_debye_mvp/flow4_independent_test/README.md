# Flow 4 independent_test precompute

This directory holds **precomputed** layered responses for the frozen
`independent_test` split (TE01–TE10). PR 10 may consume the
`case_TE*.json` files **after L1 freeze**.

This precompute is not Flow 4 execution and not an L2 gate.

- Official lagged-DLF identity (`smoke_fast_lagged_dlf`, `ft_pts_per_dec=-1`)
- All frozen templates on point receivers, six channels, total + IP increment
- W0 / W1 / W2 / W3 (W3 holdout)
- Disks (`disk_1.0`, `disk_4.0`) on the L0 shortlist only (`disk_coverage=l0_shortlist`)
- Holdout tilted coil from a linear projection of cached point six-channel fields
- No selector, no `selected_template*` files, no L2 evaluation, no 3-D

After L1 freeze, a missing P-R disk template can be added with
`--extend-disks`. L2 must refuse to score a P-R template on disks unless
those disk tasks exist.
