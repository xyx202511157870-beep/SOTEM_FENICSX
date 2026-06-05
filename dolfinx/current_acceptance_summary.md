# DOLFINx SOTEM current acceptance summary

Date: 2026-06-04

## Acceptance definition

The validation target is the 100 m grounded-wire, 50 m parallel-offset model
against the empymod semi-analytic air-earth reference.

The reported acceptance gate is:

- Strong physical responses: `Ex`, `dBzdt`, and horizontal electric vector
  norm `Eh_vector` must have max error below `5%`.
- Weak horizontal electric components: if a component reference maximum is
  below `0.1 * max(|Eh_ref|)`, its max absolute error is normalized by
  `max(|Eh_ref|)` and must be below `5%`.

The standalone pointwise relative error of `Ey` is not used as an acceptance
metric in this geometry because the `Ey` reference is close to zero and the
percentage denominator is ill-conditioned.

## Model

- Source: `(-50, 0, -0.1) -> (50, 0, -0.1)`, current `1 A`
- Source length: `100 m`
- Receiver: `(500, 50, -0.1)`
- Parallel offset: `50 m`
- Time window: `2.5e-6 s` to `1.0e-4 s`, `18` samples
- Reference: empymod finite source, primary `srcpts=65`, audit `srcpts=129`

Model sketch:

![current acceptance model geometry](figures/current_acceptance_model_geometry.png)

## Results

| Case | Ex max | dBzdt max | Eh_vector max | weak Ey scaled abs max | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Non-polarizable | `3.170741e-02` | `2.682706e-02` | `3.568113e-02` | `1.799542e-02` | pass |
| Cole-Cole IP | `3.186854e-02` | `4.093153e-02` | `3.570948e-02` | `1.618693e-02` | pass |

The `Eh_vector` values are the horizontal electric vector-error gate reported
by `verification_report.txt`.

## Runtime

| Case | Total runtime | Forward solve |
| --- | ---: | ---: |
| Non-polarizable | `536.331 s` | `517.594 s` |
| Cole-Cole IP | `583.767 s` | `561.338 s` |

## Artifacts

Run directories:

- `dolfinx/runs/noip_offset50_afterramp_recv5_mean_src65_tmin25e7_t1e4`
- `dolfinx/runs/cole_offset50_afterramp_recv5_median_src65_tmin25e7_t1e4_debye_be`

Main figures:

- `dolfinx/figures/current_acceptance_model_geometry.png`
- `dolfinx/figures/current_noip_three_component_empymod_error.png`
- `dolfinx/figures/current_cole_three_component_empymod_error.png`

Automated tests:

- `tests/test_dolfinx_verified_acceptance.py` checks the current no-IP and
  Cole-Cole run directories for both the physical response gate and weak
  horizontal-component gate.

## Current conclusion

Under the documented physical-response and weak-horizontal-component acceptance
definition, both the non-polarizable and Cole-Cole polarizable responses are
below `5%` over `2.5e-6 s` to `1.0e-4 s`.

If strict component-wise pointwise relative error is required for every saved
component, this geometry is still not accepted because `Ey` is a weak
near-zero component.
