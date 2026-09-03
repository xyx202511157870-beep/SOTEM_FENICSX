# ROADS-Debye-MVP layered protocol (frozen)

This file freezes the scientific contract for the empymod layered-first
falsification. Do not move the time window, waveforms, receivers, seeds,
candidate family, K set, or gates after seeing results.

## Scientific question

Finite Debye/Prony representations are usually chosen by minimizing Cole-Cole
complex-conductivity spectral error

`theta_sigma,K* = argmin J_sigma(theta)`.

The quantity of interest is received TEM-IP data after EM propagation,
transmitter waveform, finite-area receivers, and component projection

`theta_d,K* = argmin J_d(theta)`.

Test whether `theta_sigma,K* != theta_d,K*` with a stable, usable effect size.

### Hypotheses

- H0: on a fair identical candidate set, same K, same physics constraints, the
  spectral-best vs receiver-best difference is not large enough for a stable
  receiver-error improvement or at least 2-term qualifying-K reduction.
- H1: receiver-oriented choice on independent tasks satisfies at least one of:
  (1) same-K receiver error stably lower than spectral-best;
  (2) min K to pass receiver thresholds reduced by >= 2;
  (3) systematic bias of spectral-best on IP increment / finite turn-off /
  finite-area receivers that receiver-oriented choice clearly reduces.

If H0 cannot be rejected on cheap layered tests, STOP. Do not start 3-D.
That stop is a successful falsification, not an execution failure.

## Constitutive models

Cole-Cole uses the Pelton resistivity convention: `sigma* = 1/rho*`,
`rho0 > 0`, `0 <= m < 1`, `tau > 0`, `0 < c <= 1`.

Debye: `sigma_K*(omega) = sigma_inf - sum_k Delta_sigma_k / (1 + i omega tau_k)`,
`Delta_sigma_k >= 0`, `tau_k > 0`, `sum Delta_sigma = sigma_inf - sigma0_target`,
`sigma0_target > 0`. Reject non-passive, bad DC, or numerically unstable
candidates.

Pelton `c = 1` is a conductivity Debye at `tau * (1-m)`, not at Pelton `tau`.
Hard-DC fits use `fit_debye_passive_hard_dc` from `atem3d.adaptive_debye_mvp`
(PR 8). Gates: `delta >= -1e-12`, `rel_dc <= 1e-10`, `sigma0 > 0`, optimizer
success.

## Time window, channels, waveforms, receivers

- Time window: `1e-5 <= t <= 1e-2` s, 31 log-spaced samples. Origin = instant
  of complete current shut-off. Do not move after seeing results.
- Channels, fixed order: Hx, Hy, Hz, dBxdt, dBydt, dBzdt. Units A/m and T/s.
- Waveforms: W0 `ideal_step_off`; W1 `linear_ramp_5us`; W2 `linear_ramp_20us`
  (train/val allowed); W3 tabulated holdout
  `t = [-40, -25, -10, 0] us`, scales `[1.0, 0.80, 0.25, 0.0]` (independent
  test only); pressure W4 `linear_ramp_50us`. Exact Cole-Cole and candidates
  use identical waveform convolution settings (Gauss-Legendre order 8).
- Receivers: (1) point six-channel; (2) disk radius 1.0 m; (3) disk radius
  4.0 m; (4) three orthogonal coil-normal dB/dt projections; (5) holdout
  tilted normal `normalized([0.35, -0.20, 0.915])` as independent-test task.
  Layered production disk quadrature is a frozen 4-point in-plane square
  (`square4`) on receiver 0. AverageReceiver 36-point disks are retained as
  an audit rule only. L0 evaluates every valid template on both point
  receivers; disks are computed for the spectral-best and the top 8
  point-ranked templates per K so the finite-area check stays tractable.
- Source: asymmetric oblique finite grounded wire. Off-axis receivers so all
  six channels have real samples. Do not inflate relative error on theoretically
  zero components.

## K, candidates, spectral baselines

- Pilot K: `{4, 6, 8, 10, 12}`. Layered stage may add `{14, 16}` only if a
  method is close-but-not-quite at threshold. Never inflate K to keep a no-gap
  project alive.
- Candidate templates: 36 deterministic templates per K from
  `CandidateConfig` (2 families × 3 spans × 3 shifts × 2 densities). Families
  are Cole-Cole-tau-centered and time-window-centered. The Flow-1
  `candidate_registry.csv` freezes template identities at the protocol
  canonical Cole-Cole tau `sqrt(1e-4 * 1e-1)`. Evaluation instantiates the
  same template IDs on each case's polarizable tau. Test generates no new
  templates.
- Spectral baselines: S0 `uniform_log_frequency` and S1
  `time_window_matched_frequency_weight` on `logspace(-2, 4, 61)` Hz.
  Official B2 is whichever is better on material-spectrum metrics only
  (never chosen with receiver data).

Fair comparison: spectral-best and receiver-best are chosen from the same
`Theta_K` (same K, pole family, candidate count, passivity, hard DC,
optimizer tolerances, Cole-Cole params, empymod accuracy).

## Error and qualifying K

- Effective-data mask: `A_qc = 95th percentile of |d_ref|`; keep samples with
  `|d_ref| >= 1e-3 * A_qc`. Near-zero samples keep absolute error.
- Total-field robust error uses `max(|d_ref|, alpha A, d_floor)` with
  `alpha = 1e-2` and the library `DEFAULT_D_FLOOR`.
- IP increment `Delta d = d_IP - d_noIP` is channel NRMSE with a floor.
- Lexicographic selection: (1) fail-case rate; (2) 95th percentile of
  case-level `total_p95`; (3) 95th percentile of case-level
  `ip_increment_nrmse`; (4) worst waveform/coil/channel-group error;
  (5) median error; (6) spectral error and condition number.
- Qualifying defaults: `Q0.95(e_total) <= 1%`, `Q0.95(E_Delta) <= 3%`, no
  unexplained sign flips on effective main components, peak-time error <= one
  output dt, stable zero-crossing time error <= one output dt, any large
  waveform/receiver group fail rate <= 10%.
- `Delta K_qual = K_qual_B2 - K_qual_P-R`. Meaningful compression is
  `Delta K_qual >= 2`.

## Data splits and seeds

Do not redraw after Flow 1.

| split | n | seed |
|---|---|---|
| pilot_gap | 8 | 202609111 |
| train | 12 | 202609112 |
| validation | 6 | 202609113 |
| independent_test | 10 | 202609114 |
| layered_pressure | 4 | 202609115 |

Main ranges: `rho0` in `[20, 500]` Ohm m, `m` in `[0.05, 0.40]`,
`tau` in `[1e-4, 1e-1]` s, `c` in `[0.35, 0.90]`. Include 2-layer and
3-layer, polarizable cover/middle/basement-under-nonpolarizable.

Independent test includes unseen W3, tilted coil normal, a new source
azimuth, a new receiver offset, and interpolated parameter points (not
train-grid repeats).

Pressure set (report only, not a 3-D gate): `c` in `[0.20, 0.30]`,
`tau` in `[0.3, 3.0]` s, `m` in `[0.45, 0.55]`, 50 us slow turn-off.

Bootstrap: 2000 case-level paired resamples, seed 202609116. Case is the
sampling unit.

## Layered gates

### L0 (oracle gap, pilot only)

A same-K stable accuracy: at least one K in `{6,8,10,12}` with
(1) median of case-level receiver_error_p95 OR/B2 ratio <= 0.80;
(2) paired bootstrap 95% CI upper bound < 1.00;
(3) OR better than B2 in >= 70% of pilot cases;
(4) improvement on at least two waveforms;
(5) improvement on point AND at least one disk;
(6) at least one of H-group or dBdt-group clearly improved, the other not
systematically worse by > 10%;
(7) IP-increment p95 not worse by > 5%.

B qualifying-K: `median(K_qual_B2 - K_qual_OR) >= 2` and the difference is
nonnegative in >= 70% of pilot cases.

If A and B both fail after at most one clear-bug retry:
`STOP_LAYERED_NO_ACTIONABLE_GAP`. Do not implement a new candidate family.
Do not run independent test as a proposed-method claim. Do not run 3-D.

### L1 (selector, only if L0 passed)

P-R uses only train/val. B2 does not use receiver responses. Identical
candidate sets/constraints. Independent test unread.

### L2 (independent test, only if L1 passed)

A or B as specified in the task statement. If L0 passed but L2 failed:
`STOP_LAYERED_SELECTOR_FAILED`. If L2 passed:
`3D_AUTHORIZED_PENDING_PREFLIGHT` and stop. Do not run 3-D.

## 3-D budget note (not executed)

3-D FEniCSx is not authorized by this protocol until L2 passes, and this
execution does not run 3-D even if L2 passes. Attempting 3-D before L2 must
raise `ThreeDNotAuthorizedError`.

## Allowed final statuses

Use exactly one of:

- `STOP_LAYERED_NO_ACTIONABLE_GAP`
- `STOP_LAYERED_SELECTOR_FAILED`
- `3D_AUTHORIZED_PENDING_PREFLIGHT`
- `BLOCKED_BY_SOFTWARE_OR_RESOURCES`

Do not emit `EVIDENCE_SUPPORTS_Q1_SUBMISSION`,
`EVIDENCE_SUPPORTS_SPECIALIST_JOURNAL`, or `STOP_3D_TRANSFER_FAILED`.
Never write `Q1_READY`, `GUARANTEED_Q1`, or equivalent. Never claim
universal optimality. Never label high-K 3-D as exact Cole-Cole.

## Software reuse

Do not reimplement the Debye library. Use `atem3d.adaptive_debye_mvp`
(`passive_fit`, `candidates`, `receiver_metrics`, `bootstrap`, `io`) from
`paper/mvp-debye-lib`. Layered forwards wrap
`empymod_magnetic6` / `empymod_waveform` / `empymod_compare` /
`receivers.AverageReceiver`.
