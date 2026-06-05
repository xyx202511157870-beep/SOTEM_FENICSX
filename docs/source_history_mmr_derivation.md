# Source-History MMR Derivation Notes

This note records the equation-level constraints for the grounded-wire Debye-IP
source-primary magnetic correction.  It is intentionally narrower than the full
formulation document: the goal is to identify the remaining production term
without mixing it with empirical receiver fits.

## 1. Discrete EB Equations

The EB solver advances edge electric field `e` and face magnetic flux density
`b` with backward Euler.  For one Debye term,

```text
j_c = M_sigma_inf e - M_delta y
tau dy/dt + y = e
y^{n+1} = alpha_n y^n + beta_n e^{n+1}
alpha_n = tau/(tau + dt_n)
beta_n = dt_n/(tau + dt_n)
```

The step-off grounded source is represented by an integrated edge vector
`s_e(t)`.  The non-CPML E-form solve uses the source jump

```text
source_rhs^{n+1} = -(s_e^{n+1} - s_e^n_left)/dt_n.
```

For a long-on step-off source, the initial fields satisfy the DC balance

```text
C^T M_mu^-1 b^0 = M_sigma0 e^0 + s_0
sigma0 = sigma_inf - delta_sigma
y^0 = e^0.
```

This initial balance is verified by `ampere_source_projection` at `t=0`.

## 2. Magnetic Recovery Contract

The Biot/MMR receiver modes do not sample `b` directly.  They reconstruct
receiver magnetic field from a current-like object:

```text
q_j^{n+1} = M_sigma_inf e^{n+1} - M_delta y^{n+1}
```

or a cell-local edge-basis version of the same constitutive current.  The
positive-time impressed source field is zero for an ideal step-off, so
`H_source(t>0)` is not added by `_source_magnetic_field`.

The empirical residuals show a missing near-source magnetic contribution with
the spatial shape

```text
H_missing(t, r_k) ~= K(t) H_source_initial(r_k).
```

This is a receiver-recovery correction, not a change to the EB time-step solve.

The EB active-CPML rerun on the strict 50 m wire / 10 m offset one-Debye xy6
geometry keeps this separation intact.  The report
`outputs/debye_ip_line_exeyhz_xy6pad_cpml_current_biot_t010_1ms_validation.json`
uses the CPML config
`examples/debye_ip_line_exeyhz_xy6pad_cpml.yaml` and compares the
`0.1-1.0 ms` gates against empymod with the Debye terms read from the config.
It gives good electric-field scale (`Ex` relative L2 about
`0.081/0.038/0.081`, side `Ey` about `0.055`) and center `Hz` about `0.105`,
but the side `Hz` values remain about `3.51`.  Stored-B CPML variants reduce
the side `Hz` error to the `0.55-0.62` range while leaving the center around
`0.69-0.81`.  Thus the absorbing boundary is operational for EB, but it does
not supply the missing near-source magnetic source-history/MMR law.  The
remaining derivation below should continue to target the source/return-current
magnetic recovery operator and its Debye-driven time kernel, not CPML tuning.

## 3. Discrete Debye Source-History Basis

The useful time basis should be generated from the same backward-Euler Debye
memory update as the solver.  The diagnostic basis implemented in
`discrete_debye_history_basis` is

```text
g_0^{n+1} = alpha_n g_0^n,           g_0^0 = 1
g_p^{n+1} = alpha_n g_p^n + beta_n g_{p-1}^{n+1},  g_p^0 = 0, p >= 1.
```

For uniform small `dt`,

```text
g_0(t) -> exp(-t/tau)
g_1(t) -> (t/tau) exp(-t/tau).
```

On the strict `current_biot` residual, fitting

```text
K(t) ~= c_0 g_0(t) + c_1 g_1(t)
```

gives

```text
c_0/(mu0 delta_sigma L^2) = -6.366
c_1/(mu0 delta_sigma L^2) = -2.182
relative_l2 = 0.05631.
```

The equivalent analytic-wire and FV edge-current source bases give nearly the
same coefficients.  Therefore the static source spatial basis is not the
dominant unknown.

## 4. Transfer Constraints

The coefficients are not universal constants:

```text
strict current_biot sub1:      -6.366, -2.182
edge_basis_cell_biot sub1:     -5.409, -3.343
older current_biot sub5:      -11.389,  3.092
```

Therefore a production correction cannot hard-code a global `c_0,c_1`.  The
coefficient must be derived from the selected receiver recovery operator and
the local FV/MMR source coupling.

Any valid production term must also satisfy:

```text
1. It vanishes when there are no Debye terms.
2. It uses the same time stepping as the Debye memory variables.
3. It is independent of empymod residual fitting.
4. It transfers across receiver offsets, mesh refinement, and boundary sweeps.
5. It preserves the existing no-IP current-Biot behavior.
```

### Residual Target for IP-Only Derivation

The sampled-report diagnostics distinguish four useful targets:

```text
reference_delta = IP_ref - noIP_ref
numerical_delta = IP_num - noIP_num
ip_residual     = IP_ref - IP_num
noip_residual   = noIP_ref - noIP_num
delta_residual  = reference_delta - numerical_delta
                = ip_residual - noip_residual.
```

`ip_residual` measures the total error of the IP run, but it contains the
existing no-IP H/J magnetic baseline error.  A production source-history term
that vanishes when `delta_sigma = 0` cannot be derived from that contaminated
target.  The cleaner target is therefore `delta_residual`, now exposed by the
source-primary and source-history matrix CLIs.

For the source-centered H/J `current_biot` line, the old `ip_residual` target
made the per-time source-moment coefficient change sign at the earliest gates
and needed order-6 BE cascades with normalized condition number about `3.6e7`
to reach relative L2 `0.049`.  The corrected `delta_residual` target removes
that no-IP common-mode component:

```text
static source-moment {0,2} projection error: 2.4e-4
BE order 1 all-window relative L2:          0.194
BE order 2 all-window relative L2:          0.142
BE order 3 all-window relative L2:          0.090
BE order 4 all-window relative L2:          0.053
BE order 5 all-window relative L2:          0.0317
BE order 6 all-window relative L2:          0.0202
order-6 normalized condition number:        1.02e5
```

The saved reports are:

```text
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_orders1to4.json
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_orders5to6_all.json
```

Both reports now include `spatial_time_series.history_basis_fits`, which fits
the per-time static source-moment coefficient traces directly to the BE history
columns.  This removes the receiver matrix from the second step of the audit.
For the source-centered `delta_residual` target, the trace fits closely match
the full receiver-space fits: order 1 gives aggregate relative L2 `0.193`
(`0.190/0.227` per `{s_0, xi^2 s_0}` trace), with normalized coefficient table
`[[6.758, 2.139], [3.415, 0.520]]`; order 4 gives `0.0533`; order 5 gives
`0.0318`; and order 6 gives `0.0204`.  The trace-fit normalized condition
number rises from `3.72` at order 1 to `2.48e4` at order 6, while the
full receiver-space order-6 condition number is `1.02e5`.  This confirms that
the static `{0,2}` source-face moments are not the limiting factor; the missing
production term is the non-fitted H/J time-kernel/coefficient coupling that
should generate these traces from the discrete grounded-source Debye/MMR
equations.

The scan reports can now also evaluate prescribed coefficients in this same
trace space through `spatial_time_series.prescribed_history_basis`.  This is the
intended non-fitted-candidate hook: a derived coefficient table should be passed
through the prescribed path and checked both in receiver space and in coefficient
trace space.  The sample report
`outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_order1_trace_prescribed.json`
uses the order-1 diagnostic trace coefficients as a round-trip check; it records
receiver-space prescribed relative L2 `0.19370` and trace-space prescribed
relative L2 `0.19317`, with normalized table
`[[6.758, 2.139], [3.415, 0.520]]`.  Future production derivations should
replace those fitted numbers with coefficients computed from the FV/HJ Debye
source-history operator and use the same two errors as the acceptance audit.

The matrix and scan CLIs now also accept dimensionless prescribed coefficients
through

```text
--prescribed-normalized-coefficients <values>
```

These values are multiplied internally by
`mu * delta_sigma * source_length^2` from the loaded simulation before the same
receiver-space and trace-space audits are run.  This avoids hand-converting
candidate coefficient tables during non-fitted derivation tests.  The EB and H/J
runtime `magnetic_recovery_source_history` blocks now expose the same convention
as `normalized_coefficients` for `prescribed_source_moments` and
`driven_recovery_source_moments`; at runtime the factor is
`mu * sum(delta_sigma_source) * source_length^2`, so a zero Debye contrast still
removes the IP-only correction.

The rounded material-source candidate

```text
[[6, 2],
 [3, 0.5]] * mu * delta_sigma * L^2
```

was evaluated with this normalized path in
`outputs/source_history_matrix_hj_delta_residual_material_source_6_2_3_05_normalized_candidate.json`.
Refreshed against the current source-centered sub5 sampled reports, it gives
receiver-space prescribed relative L2 `0.197558` and trace-space relative L2
`0.203709`.  This remains above the current order-1 fitted table
`[[5.662, 1.165], [4.831, 1.797]]`, whose receiver-space and trace-space errors
are about `0.18297` and `0.18185`.

The companion window scan
`outputs/source_history_matrix_scan_hj_delta_residual_material_source_6_2_3_05_normalized_windows.json`
is more restrictive:

```text
all:            fit 0.182969, prescribed 0.197558
0.10-0.30 ms:  fit 0.105365, prescribed 0.319363
0.40-0.70 ms:  fit 0.000665, prescribed 0.126789
0.80-1.00 ms:  fit 0.000224, prescribed 0.020991
```

The independently fitted normalized order-1 tables drift from roughly
`[0.017, -0.993; 32.201, 12.191]` in the early window to
`[8.683, 2.323; -0.0066, -0.072]` in the late window.  Therefore
`[[6, 2], [3, 0.5]]` is a useful low-dimensional constraint on the material
source shape, but it is not a transferable production time-kernel law.

The high-frequency no-IP companion config
`examples/hj_highfreq_noip_line_exeyhz_source_centered_axis_aligned.yaml`
uses the same source-centered H/J mesh but sets the layer conductivities to the
Debye `sigma_infinity` values (`0.012/0.036 S/m`) without memory variables.  It
was used to test whether a finite-volume high-minus-low no-IP conductivity
sensitivity can stand in for the missing IP source-history trace.  The data-only
run and sampled report are

```text
outputs/hj_highfreq_noip_line_exeyhz_source_centered_axis_aligned_data_only.h5
outputs/hj_highfreq_noip_line_exeyhz_source_centered_axis_aligned_t010_1ms_samples_report.json
```

The source-history matrix reports

```text
outputs/source_history_matrix_hj_high_minus_low_noip_numerical_delta_source_moments.json
outputs/source_history_matrix_hj_high_minus_low_noip_reference_delta_source_moments.json
outputs/source_history_matrix_hj_high_minus_low_noip_delta_residual_source_moments.json
```

and the summary
`outputs/source_history_hj_high_low_noip_sensitivity_comparison.json` reject
this shortcut as a direct production law.  The high-minus-low noIP reference
delta has its own order-1 trace fit relative L2 `0.2867`, with normalized
coefficients `[[9.096, 4.673], [-9.987, -4.985]]`; compared directly to the
missing H/J `delta_residual` trace, even the best common scale leaves relative
L2 `0.6778`.  The numerical high-minus-low trace is worse (`0.8169` against the
missing trace), and the high-minus-low residual trace is essentially rejected
(`0.9736`).  Thus the remaining source-history term is not just the no-IP
receiver response to switching the background conductivity from low-frequency
to high-frequency values.

### Zero-Initial Rise-Decay Trace Kernel

The clean `delta_residual` coefficient trace is not shaped like a homogeneous
initial Debye memory.  The normalized `{s_0, xi^2 s_0}` trace starts small at
`0.10 ms`, rises by about `0.20-0.30 ms`, then decays toward `1.00 ms`.  This
motivates the discrete zero-initial diagnostic basis

```text
k_f^n = g_slow^n - g_fast^n
g_tau^{n+1} = tau/(tau + dt_n) g_tau^n,  g_tau^0 = 1
fast_tau < slow_tau.
```

`k_f^0 = 0`, so it represents a build-up/recovery convolution rather than an
initial-condition decay.  The helper
`discrete_relaxation_difference_basis` returns this column on the exact BE time
grid, and the CLI `atem3d.source_history_trace_kernel_cli` fits the saved
`spatial_time_series` coefficient traces without touching the runtime physics.
It requires the matching HDF5 result so the positive-time samples are matched to
the original time nodes, not to a coarsened grid inferred from sampled gates.

The same shape can be written as a more physical driven recovery state:

```text
z^0 = 0
z^{n+1} = a_R^n z^n + b_R^n g_slow^{n+1}
a_R^n = tau_R/(tau_R + dt_n)
b_R^n = dt_n/(tau_R + dt_n)
```

Here `g_slow` is the Debye source-memory decay and `z` is a local MMR/source
recovery state with response time `tau_R`.  On uniform time steps,

```text
z^n = tau_slow/(tau_slow - tau_R) * (g_slow^n - g_R^n).
```

The proportionality factor is absorbed into the spatial coefficient, so the
trace fit using `g_slow-g_fast` is equivalent to fitting this driven state.
This form is the better production-law skeleton: the unknown should be a local
finite-volume recovery operator that supplies one or more `tau_R`-like modes
and coupling amplitudes, not a hand-picked difference of exponentials.
`discrete_driven_relaxation_basis` is tested against this BE recursion and the
uniform-step equivalence.

The source-centered H/J `current_biot` `Hz delta_residual` report

```text
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_order1_trace_prescribed.json
```

was scanned with `slow_tau=1 ms` and fast candidates from `0.02` to `0.50 ms`.
The result is saved in

```text
outputs/source_history_hj_rise_decay_kernel_scan.json
```

The best single rise-decay column is

```text
fast_tau = 0.11 ms
trace relative_l2 = 0.12183
per-trace relative_l2 = [0.11749, 0.16510]
coefficients/(mu0 delta_sigma L^2) = [[9.319, 2.677]]
```

This improves on both the order-1 same-pole BE trace fit (`0.18185`) and the
rounded material-source candidate (`0.20371` in trace space).  The improvement
is structural: the candidate has the required zero initial value and early
rise, which the nonzero-initial `g_0, g_1` basis can only approximate by
cancellation.

The same report also contains an all-fast-tau multi-column fit.  It reaches
trace relative L2 `3.87e-4`, but the column-normalized condition number is
about `2.0e12` and the largest normalized coefficient is about `1.6e9`.  This
is a cancellation basis and should not be promoted.  The useful conclusion is
therefore narrower: the production H/J MMR/source-history law probably needs a
causal source-recovery convolution with a fast build-up time near `0.1 ms`, not
another direct projection of an initial H/J state or a fixed high/low
conductivity sensitivity.

The trace-kernel CLI also supports `--time-min` and `--time-max` so the same
rise-decay hypothesis can be audited on subwindows.  The four reports

```text
outputs/source_history_hj_rise_decay_kernel_all.json
outputs/source_history_hj_rise_decay_kernel_early_010_030ms.json
outputs/source_history_hj_rise_decay_kernel_middle_040_070ms.json
outputs/source_history_hj_rise_decay_kernel_late_080_100ms.json
```

are summarized in
`outputs/source_history_hj_rise_decay_kernel_window_summary.json`:

```text
all 0.10-1.00 ms:  best fast_tau = 0.11 ms, L2 = 0.12183
0.10-0.30 ms:      best fast_tau = 0.30 ms, L2 = 0.14178
                   fixed 0.11 ms L2 = 0.18962
0.40-0.70 ms:      best fast_tau = 0.02 ms, L2 = 0.02023
                   fixed 0.11 ms L2 = 0.03202
0.80-1.00 ms:      best fast_tau = 0.02 ms, L2 = 0.00647
                   fixed 0.11 ms L2 = 0.00692
```

This transfer check is important.  A single rise-decay column is clearly a
better all-window shape than the nonzero-initial `g_0/g_1` basis, but the
effective fast time drifts by window and the multi-fast-tau fits become
rank-deficient/cancellation dominated.  Therefore the next production
derivation should not hard-code `fast_tau = 0.11 ms`.  It should instead derive
a local H/J MMR/source-recovery convolution whose projected source-moment trace
has this zero-initial build-up behavior and whose effective coefficients vary
with the recovery path and time window in a controlled finite-volume way.

The same clean-target workflow was repeated for the source-centered
`current_biot` run with `magnetic_recovery_subdivisions=5`.  The matrix scan

```text
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_sub5_delta_residual_hz_line_source_moments_even02_orders1to4.json
```

uses the sub5 IP/noIP sampled reports and `--subdivisions 5` in the receiver
matrix.  It gives the following trace-space BE-history audit:

```text
order 1 trace L2 = 0.18174
  coefficients/(mu0 delta_sigma L^2) = [[5.551, 1.113],
                                        [4.738, 1.729]]
order 2 trace L2 = 0.11564
order 3 trace L2 = 0.06045
order 4 trace L2 = 0.02729
```

The sub5 rise-decay scans

```text
outputs/source_history_hj_rise_decay_kernel_sub5_all.json
outputs/source_history_hj_rise_decay_kernel_sub5_early_010_030ms.json
outputs/source_history_hj_rise_decay_kernel_sub5_middle_040_070ms.json
outputs/source_history_hj_rise_decay_kernel_sub5_late_080_100ms.json
outputs/source_history_hj_rise_decay_kernel_sub5_window_summary.json
```

show:

```text
all 0.10-1.00 ms:  best fast_tau = 0.12 ms, L2 = 0.10088
                   coefficients/(mu0 delta_sigma L^2) = [[8.780, 2.178]]
0.10-0.30 ms:      best fast_tau = 0.50 ms, L2 = 0.11209
                   fixed 0.12 ms L2 = 0.17917
0.40-0.70 ms:      best fast_tau = 0.02 ms, L2 = 0.00415
                   fixed 0.12 ms L2 = 0.01995
0.80-1.00 ms:      best fast_tau = 0.02 ms, L2 = 0.00049
                   fixed 0.12 ms L2 = 0.00112
```

This is a useful transfer result.  The all-window effective build-up time stays
near `0.1 ms` when the Biot recovery quadrature is refined from sub1 to sub5,
and the fitted rise-decay coefficients remain `O(10)` in normalized units.
However the early-window preferred fast time shifts from `0.30 ms` to the
largest checked `0.50 ms`, while the middle and late windows again collapse to
the fastest checked mode.  The production term is therefore not a universal
scalar recovery time.  It should be a local recovery operator with a spectrum of
fast modes; the compact one-mode rise-decay fit is only the all-window
low-dimensional shadow of that operator.

The same clean-target experiment was repeated with the H/J
`face_basis_cell_biot` receiver reconstruction and subcell quadrature `5`.  This
path reconstructs face-current basis functions inside each cell before the
Biot-Savart integration, so it is a stronger check that the missing law is not
an artifact of the original cell-current averaging.  The reports are:

```text
outputs/hj_debye_ip_line_exeyhz_source_centered_axis_aligned_face_basis_cell_biot_sub5_t010_1ms_samples_report.json
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_face_basis_cell_biot_sub5_delta_residual_hz_line_source_moments_even02_orders1to4.json
outputs/source_history_matrix_fit_hj_source_centered_axis_aligned_face_basis_cell_biot_sub5_delta_residual_hz_line_source_moments_even02_order1.json
```

The one-shot matrix CLI report uses the matrix-free static-response path rather
than constructing the full dense `face_basis_biot_matrix`.  Its order-1 result
matches the scan:

```text
static source-moment {0,2} projection error: 2.327e-4
static response rank/shape:                  2 / [3, 2]
order 1 receiver-space L2:                   0.19671
order 1 trace-space L2:                      0.19485
coefficients/(mu0 delta_sigma L^2):          [[5.572, 0.881],
                                              [5.368, 2.096]]
```

The higher BE history orders reduce the trace residual in the same way as the
`current_biot` path:

```text
order 1 trace L2 = 0.19485
order 2 trace L2 = 0.12375
order 3 trace L2 = 0.06499
order 4 trace L2 = 0.02931
```

The face-basis rise-decay window summary is saved in

```text
outputs/source_history_hj_rise_decay_kernel_face_basis_cell_biot_sub5_window_summary.json
```

and shows:

```text
all 0.10-1.00 ms:  best fast_tau = 0.15 ms, L2 = 0.11279
                   coefficients/(mu0 delta_sigma L^2) = [[9.672, 2.248]]
0.10-0.30 ms:      best fast_tau = 0.50 ms, L2 = 0.13939
                   fixed 0.15 ms L2 = 0.19185
0.40-0.70 ms:      best fast_tau = 0.02 ms, L2 = 0.00255
                   fixed 0.15 ms L2 = 0.03095
0.80-1.00 ms:      best fast_tau = 0.10 ms, L2 = 0.00037
                   fixed 0.15 ms L2 = 0.00244
```

This is not a simpler production law.  The face-basis receiver improves or
reshapes the raw magnetic recovery, but the coefficient traces still require a
rise-decay response whose effective fast time depends on the time window.  A
multi-fast-tau fit can again drive the trace L2 to `2.09e-4`, but only with
column-normalized condition number about `1.63e12` and normalized coefficients
up to `7.98e8`.  This is cancellation evidence, not an acceptable FV/MMR source
history formula.

The trace-kernel CLI now has a second basis option for keeping the offline
audit in the same variables as the runtime hook:

```text
--basis-kind driven_relaxation
```

This replaces the diagnostic `g_slow - g_fast` column by the BE state

```text
z^{n+1} = a_R z^n + b_R g_slow^{n+1},  z^0 = 0,
```

which is the same history used by `driven_recovery_source_moments`.  On the
current uniform time grid it changes only the coefficient scale, not the fit
quality.  The sub5 all-window reports

```text
outputs/source_history_hj_driven_relaxation_kernel_current_biot_sub5_all.json
outputs/source_history_hj_driven_relaxation_kernel_face_basis_cell_biot_sub5_all.json
```

give:

```text
current_biot sub5:       best response_tau = 0.12 ms, L2 = 0.10088
                         coefficients/(mu0 delta_sigma L^2) = [[7.727, 1.917]]
face_basis_cell_biot:    best response_tau = 0.15 ms, L2 = 0.11279
                         coefficients/(mu0 delta_sigma L^2) = [[8.221, 1.911]]
```

This is a useful bookkeeping improvement for the derivation: future non-fitted
coefficients can be compared directly in runtime-driven-state units.  It does
not change the physical conclusion that the missing term is the operator that
generates the response time and coefficient distribution.

### Runtime Driven-Recovery Hook

The offline driven-state skeleton is now available in the diagnostic runtime
source-history hook:

```yaml
magnetic_recovery_source_history:
  kind: driven_recovery_source_moments
  driver_tau: 0.001
  response_tau: 0.00011
  source_moment_degrees: [0, 2]
  coefficients: [...]
```

The runtime evaluates

```text
Delta H_R(t_n) = z^n sum_s a_s B_R v_s
```

where `z^n` is the BE driven recovery state and `B_R v_s` is the same FV/MMR
receiver matrix applied to the source-moment vectors.  This is still
diagnostic-only: the coefficients in the first example are the all-window trace
fit, not a derived local operator.

The example

```text
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_driven_recovery.yaml
```

uses the sub1 all-window one-mode coefficients
`[5.855339e-05, 1.681984e-05]`, equivalent to normalized
`[[9.319, 2.677]]`, with `driver_tau=1 ms` and `response_tau=0.11 ms`.  Its
data-only result and empymod validation are

```text
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_driven_recovery_data_only.h5
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_driven_recovery_t010_1ms_validation.json
outputs/hj_driven_recovery_runtime_validation_summary.json
```

The `Hz` relative L2 comparison over `0.10-1.00 ms` is:

```text
uncorrected current_biot:  [0.588, 0.650, 0.589]
order-6 fitted BE hook:   [0.199, 0.204, 0.200]
one-mode driven hook:     [0.194, 0.205, 0.195]
```

The driven hook therefore provides nearly the same runtime correction quality
as the much higher-order fitted BE diagnostic while using a physically
interpretable zero-initial recovery state.  This does not close the derivation,
but it gives the next non-fitted FV/MMR coefficient law an immediate acceptance
path: compute `a_s` and any local recovery modes from operators, plug them into
`driven_recovery_source_moments`, and compare against the same empymod gate.

The sub5 transfer check is now reproducible without re-solving the H/J system.
The source-history hook only changes magnetic receiver sampling, so an existing
uncorrected receiver-only HDF5 can be postprocessed:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_history_postprocess_cli `
  outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_data_only.h5 `
  --config examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_driven_recovery.yaml `
  -o outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_driven_recovery_postprocessed_data_only.h5
```

This avoids the PARDISO memory expansion failure observed during a full sub5
re-run.  The summary
`outputs/hj_driven_recovery_runtime_subdivision_summary.json` records:

```text
sub1 uncorrected Hz:          [0.588, 0.650, 0.589]
sub1 driven Hz:               [0.194, 0.205, 0.195]
sub5 uncorrected Hz:          [0.414, 0.476, 0.414]
sub5 driven postprocessed Hz: [0.261, 0.291, 0.262]
```

The improvement confirms that the runtime/postprocess path has the expected
sign and scale, while the weaker sub5 transfer keeps the main conclusion
unchanged: the production correction must derive local recovery modes and
source-moment coefficients from FV/MMR operators rather than reuse fitted
residual coefficients.

The source-moment scans now also honor `--source-vector`, so the same compact
moment audit can be run on `wire`, `dc_total_current`, or
`dc_polarization_current` bases instead of silently using the wire vector.  On
the source-centered H/J `delta_residual` target, the reports

```text
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_dc_total_source_moments_even02_orders1to4.json
outputs/source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_dc_polarization_source_moments_even02_orders1to4.json
```

show that these more distributed FV current bases do not reduce the receiver
residual relative to the wire source moments: the all-window receiver-space L2
sequence remains `0.1937`, `0.1422`, `0.0903`, `0.0532` for orders 1-4.  Their
static projection errors are also the same to the displayed precision
(`2.4e-4`).  The dc-polarization basis improves column-normalized conditioning
but requires very large normalized coefficients (`O(10^2-10^3)`), so it is not
the missing local law by itself.  This rules out another shortcut: replacing the
wire source moment with a DC total-current or initial-polarization-current
moment changes coordinates, not the unresolved H/J time-kernel coupling.

As an upper-bound integration check, the order-6 fitted coefficients from the
second report are wired into
`examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta_residual_order6.yaml`.
The validation report
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta_residual_order6_t010_1ms_validation.json`
reduces `Hz` relative L2 from the uncorrected source-centered IP values
`0.588/0.650/0.589` to about `0.199/0.204/0.200`.  The remaining error is close
to, but still above, the no-IP `current_biot` baseline (`0.161/0.153/0.161`).
The report metadata marks the source-history block as `diagnostic_only: true`.

This does not make the fitted coefficients production physics.  It changes the
derivation target: the non-fitted local FV/MMR operator should reproduce the
smooth, IP-only `delta_residual` source-moment traces, not the earlier
sign-changing total IP-run residual.

## 5. Excluded Candidates

### Fixed Exponential Multiplier

The empymod-only tau transfer scan rejects a universal
`tau_kernel = c tau`.  The best checked kernel changes with Debye tau, and the
empirical kernel can change sign for larger tau values.

### Static Source Basis Mismatch

Using the FV edge-current source basis instead of analytic `H_wire` changes the
strict residual BE coefficients only slightly:

```text
wire basis:         -6.366, -2.182
edge_current basis: -6.340, -2.176
```

The missing term is not primarily a line-source spatial-discretization error.

### H/J Source Sign or Initial MMR Balance

The H/J convention is now explicitly checked as

```text
s_face = -source.initial_face_vector(mesh)
j = C h - s_face
C h0 = j0 + s_face.
```

The DC return current also satisfies `D (j0 + s_face) = 0`.  The strengthened
unit test `test_hj_mmr_initial_magnetic_field_satisfies_stabilized_system`
checks both balances to machine precision and verifies that the opposite source
sign leaves an `O(1)` residual.  The `atem3d-ampere-source-projection` CLI can
now read H/J `e/h` HDF5 results and project

```text
r_HJ = C h - q(E, y) - s_face
```

onto the signed face-source vector, using the same face-projected constitutive
current as the H/J solver.  This makes source-sign and initial-MMR errors
auditable without empymod fitting.  On the saved full-field pair

```text
outputs/hj_debye_xy6_dt001ms_100step_pardiso_signfix.h5
outputs/hj_noip_xy6_dt001ms_100step_pardiso_signfix.h5
```

the generated report
`outputs/hj_ampere_source_projection_debye_minus_noip_be_basis.json` gives
H/J IP-minus-noIP normalized source-projection coefficients between about
`-2.5e-13` and `3.9e-13`; the fitted BE `g_0,g_1` coefficients are about
`-1.3e-13` and `-1.0e-13`.  These are orders of magnitude below the
`O(1-10)` source-history coefficients extracted from the H/J `Hz`
`delta_residual`, so the missing term is not hidden in the H/J Ampere residual.
It has to enter through the receiver-side MMR/source-history recovery operator.

The source-history matrix CLI can now consume that kind of non-fitted candidate
directly:

```text
--prescribed-coefficients-file <candidate.json>
--prescribed-coefficients-key discrete_basis_fit.coefficients
--prescribed-spatial-trace-index 0
```

The file/key pair is read as a candidate BE history coefficient series and, when
`--prescribed-spatial-trace-index` is supplied, expanded into one selected
source-moment trace with all other traces set to zero.  The audit
`outputs/source_history_matrix_hj_delta_residual_ampere_projection_candidate.json`
maps the H/J Ampere-projection `g_0,g_1` coefficients onto the degree-zero
source-face moment for the source-centered `Hz delta_residual` target.  It gives
receiver-space prescribed relative L2 `1.0` and trace-space prescribed relative
L2 `1.0`, while the optimal order-1 trace fit remains `0.193`.  This is a
reproducible rejection of the Ampere-projection candidate inside the same
prescribed-coefficient path that future derived FV/MMR candidates must pass.

There is now also a public vector-to-source-moment projection primitive:
`source_history_operator.project_vector_to_spatial_basis`.  Given a local FV
candidate vector `v`, source/spatial moment vectors `S`, and the selected MMR
receiver matrix `B_R`, it can compute coefficients either in DOF space
(`projection='dof_l2'`) or in receiver space (`projection='receiver_l2'`):

```text
receiver_l2:  min_a || B_R S^T a - B_R v ||_2
dof_l2:       min_a || S^T a - v ||_2
```

The runtime initial-polarization diagnostic now uses this same operator instead
of a private projection copy.  This is the intended bridge for the remaining
derivation: once the FV/HJ MMR analysis produces a candidate local source vector
or compact vector family, its coefficients can be computed without empymod and
then passed through the prescribed JSON/file path above.

The matrix CLI now has the same initial-polarization candidate built in:

```text
--prescribed-candidate initial_polarization
--prescribed-candidate-projection receiver_l2|dof_l2
```

It computes the local vector `-delta_sigma y0`, projects it through
`project_vector_to_spatial_basis`, places the coefficients in the `g_0` row, and
zeros higher BE cascades.  On the source-centered H/J `Hz delta_residual` audit,
the receiver-space projection report
`outputs/source_history_matrix_hj_delta_residual_initial_polarization_candidate.json`
records `requires_ip: true` and gives receiver-space prescribed relative L2
`63.2` and trace-space prescribed relative L2 `364`.  Its normalized `g_0`
coefficient row is about
`[942, -2811]`, far outside the `O(1-10)` fitted trace coefficients.  The
matching DOF-space projection report
`outputs/source_history_matrix_hj_delta_residual_initial_polarization_candidate_dof_l2.json`
is also rejected, with receiver-space relative L2 `81.1`, trace-space relative
L2 `281`, and normalized coefficients about `[-278, 2272]`.  Thus this
candidate is rejected through the same matrix/trace path for both available
projection norms.

The next non-fitted variant enforces the discrete continuity equation before
the source-moment projection.  For the raw initial Debye current
`j_p = -delta_sigma y0`, it solves the H/J face-current projection

```text
D (j_p - M_rho_inf^-1 G phi) = 0,
```

where `D = diag(V) face_divergence`, `G = D^T`, and `M_rho_inf` is the
face-resistivity mass built from `sigma_infinity`.  The resulting solenoidal
current is exposed as
`charge_conserving_initial_polarization_source_moments` and can be applied with
the receiver-only postprocess CLI using
`examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_charge_conserving_initial_polarization.yaml`.
The same vector is now available in the matrix CLI as
`--prescribed-candidate charge_conserving_initial_polarization`, which writes
`requires_ip: true` in the candidate metadata.  The receiver-space projection
report
`outputs/source_history_matrix_hj_delta_residual_charge_conserving_initial_polarization_candidate.json`
is rejected with prescribed relative L2 `66.9` (`382` in trace space), and the
DOF projection report
`outputs/source_history_matrix_hj_delta_residual_charge_conserving_initial_polarization_candidate_dof_l2.json`
is still far too large at receiver-space L2 `33.7` and trace-space L2 `77.2`.
The normalized `g_0` rows are about `[967, -2952]` and `[41.8, 629]`,
respectively, so the candidate is orders of magnitude away from the fitted
clean-delta source-moment traces.

This makes the candidate stricter than raw `-delta_sigma y0`, but it is still
rejected by the empymod gate as well: the validation report
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_charge_conserving_initial_polarization_t010_1ms_validation.json`
gives `Hz` relative L2 about `25.46/28.46/25.46`, and the opposite-sign control
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_charge_conserving_initial_polarization_opposite_t010_1ms_validation.json`
still gives about `24.35/29.68/24.35`.  Thus the missing source-history/MMR
operator is not just the charge-conserving projection of the initial
polarization current onto source moments.

A tempting source-path variant was also checked and rejected.  It treats the
impressed grounded-wire face source `s_0` as if it had a low-frequency material
electric field memory,

```text
E_source0 = M_rho0 s_0 / M_unit,
j_p_source = delta_sigma E_source0,
```

then projects `-j_p_source` onto the same `{0,2}` source-face moments.  This is
saved in
`outputs/source_history_source_path_memory_candidate_trace_audit.json`.  The
candidate is far too large: the normalized coefficient is about
`[-3.18e4, 0]` in `mu*delta_sigma*L^2` units, while the target trace starts at
`[0.82, 0.15]` and ends near `[3.15, 0.79]`.  Using either the `1 ms` Debye
decay or the tested zero-initial rise-decay kernels leaves relative L2 in the
range `3.26e3-3.65e3`.  Therefore the missing term is not obtained by treating
the impressed source path itself as an ordinary polarizable material current.

The related DC-current-difference shortcut was also checked offline.  It uses
the same grounded source to solve two H/J DC current systems, one with
`sigma_0` and one with `sigma_infinity`, then projects `j_inf - j_0` or
`j_0 - j_inf` onto the `{0,2}` face-source moments and advances that vector
with the `tau=1 ms` BE `g_0` history.  The receiver-space projected
coefficients are only about `[-1.18e-9, 2.03e-9]` (or the opposite sign), and
the validation reports
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_dc_current_difference_high_minus_low_t010_1ms_validation.json`
and
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_dc_current_difference_low_minus_high_t010_1ms_validation.json`
leave `Hz` at essentially the uncorrected values: about
`0.588/0.650/0.589`.  Thus the missing term is not the difference between
low-frequency and high-frequency DC grounded-source current systems either.

### Direct H/J State Vectors

A full-field source-centered H/J run was generated at
`outputs/hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_full.h5`
so that the stored face electric field can be replayed through the Debye memory
update without empymod fitting.  The audit
`outputs/source_history_hj_state_candidate_projection.json` projects several
direct internal-state current candidates onto the same `{source_face_moment:0,
source_face_moment:2}` basis:

```text
delta*y
-delta*y
delta*e
-delta*e
delta*(y - e)
delta*(e - y)
delta*(y - y0)
delta*(y0 - y)
delta*(e - e0)
delta*(e0 - e)
```

None is acceptable as a common source-history correction.  With one common
scalar applied to both traces, the best candidates still have relative L2 about
`0.998`.  Allowing independent empirical scales by source moment lowers some
single-trace errors, for example `delta*(e-e0)` gives about `0.274` on
`source_face_moment:0` and `0.330` on `source_face_moment:2`, but the required
scales have opposite signs (`+5.9e-3` and `-5.6e-4`) and the raw normalized
coefficients are `O(10^2-10^3)`, not `O(1-10)`.  This rules out another tempting
shortcut: the missing term is not a direct projection of the local H/J Debye
memory or electric-field state.  It has to be a derived receiver/source
recovery convolution or local MMR operator.

### Raw Ampere Residual

The raw stored-field residual

```text
r_A = C^T M_mu^-1 b - q_j - s_e
```

is far too large as a magnetic correction.  Projecting it onto the initial
source vector does not fix the scale:

```text
<r_A, s_0>/<s_0, s_0> /(mu0 delta_sigma L^2) ~= -5e5.
```

Subtracting a matching no-IP projection cancels this large common-mode term and
leaves coefficients near zero:

```text
Debye - noIP Ampere source projection:
g_0,g_1 coefficients ~= 7.5e-11, 5.6e-9.
```

Thus the missing `O(1-10)` source-history term is neither the raw Ampere
residual nor its IP-minus-noIP source-vector projection.

### Simple Chargeability Rescaling

After introducing the IP-only target
`delta_residual = reference_delta - numerical_delta`, the numerical IP magnetic
increment was tested as a possible non-fitted basis:

```text
H_test = H_noIP_num + f (H_IP_num - H_noIP_num).
```

For the source layer in the current source-centered case,
`sigma_inf = 0.012 S/m`, `delta_sigma = 0.002 S/m`, and
`sigma0 = 0.010 S/m`, so the low-frequency chargeability ratio is
`m = delta_sigma / sigma0 = 0.2`.  The simple rule `f = 1 - 2m = 0.6` gives
`Hz` relative L2 about:

```text
x=-20 m: 0.226
x=0 m:   0.194
x=20 m:  0.227
```

This is a large improvement over the uncorrected IP run, but it is worse than
the fitted order-6 `delta_residual` source-history diagnostic and does not
reach the no-IP magnetic baseline.  A pure chargeability scaling of the
numerical IP increment is therefore useful evidence that the H/J receiver-side
IP increment is over-strong, but it is not a complete production formula.

The same factor must also not be moved blindly into the local cell-IP receiver
law.  Running the H/J `face_basis_cell_biot` recovery with
`magnetic_recovery_polarization_scale = 0.6` and subcell quadrature `5` gives
the report
`outputs/hj_debye_ip_line_exeyhz_source_centered_face_basis_cell_biot_polscale06_sub5_t010_1ms_validation.json`.
Its `Hz` relative L2 is about `9.82/8.00/9.81`, much worse than both
`current_biot` and the unscaled face-basis-cell recovery.  Therefore the useful
`1 - 2m` data-level behavior is not a local multiplier on the Debye memory
current; the missing term still has to be a source-history/MMR coupling.

## 6. Remaining Production Derivation Target

The remaining unknown is a local operator coefficient mapping the grounded
source step-off history into the receiver-side MMR/Biot recovery:

```text
H_corr^R(t, r_k)
  = sum_m mu0 delta_sigma_m L^2
      [a_{R,m,0} g_{m,0}(t) + a_{R,m,1} g_{m,1}(t) + ...]
      H_source^R(r_k).
```

Here `R` denotes the recovery path (`current_biot`, `edge_basis_cell_biot`,
etc.).  The empirical evidence suggests that the basis functions are correct,
but the coefficients `a_{R,m,p}` must be computed from the FV source projection
and the chosen MMR recovery operator.  The next implementation step should
derive and test that coefficient calculation on a small mesh where the source
edge vector, current recovery operator, and Biot receiver matrix can be written
as explicit linear maps.

The first explicit maps are now available for the runtime cell-current
`current_biot`, strict edge-current, edge-basis, and cell-local IP Biot paths:

```text
H_current = B_current q_edge
H_edge(r_k) = B_edge(r_k) q_edge
H_basis(r_k) = B_basis(r_k) u_edge
H_cell(r_k) = B_ohm(r_k) e_edge + sum_i B_mem_i(r_k) y_i
```

where `cell_current_biot_matrix(mesh, locations, subdivisions)` maps the same
edge-current moments through the runtime `current_biot` recovery
(`M_1^-1`, `average_edge_to_cell_vector`, then cell-current Biot integration),
`edge_current_biot_matrix(mesh, locations)` returns
`B_edge[k, component, edge]`, and
`edge_basis_biot_matrix(mesh, locations, subdivisions)` returns the analogous
small-mesh diagnostic matrix for edge-basis reconstructed currents.
`edge_basis_cell_ip_biot_matrices` returns the ohmic matrix and one signed
memory matrix per Debye term for the cell-local IP recovery path.  The existing
`biot_savart_h_from_edge_current_moments` uses `B_edge` internally, and the test
suite verifies

```text
einsum("kce,e->kc", B_edge, q_edge)
  == biot_savart_h_from_edge_current_moments(mesh, q_edge, locations).

einsum("kce,e->kc", B_current, q_edge)
  == biot_savart_h_from_cell_currents(
       mesh,
       reshape(A_ec M_1^-1 q_edge, (n_cells, 3), order="F"),
       locations).

einsum("kce,e->kc", B_basis, u_edge)
  == biot_savart_h_from_edge_basis_currents(mesh, u_edge, locations).

einsum("kce,e->kc", B_ohm, e_edge)
  + sum_i einsum("kce,e->kc", B_mem_i, y_i)
  == biot_savart_h_from_edge_basis_cell_ip_currents(...).
```

The same explicit receiver-matrix scaffold now exists for the H/J face-current
placement:

```text
H_face_current = B_face_current q_face
H_face_basis(r_k) = B_face_basis(r_k) u_face
```

`face_current_biot_matrix(mesh, locations, subdivisions)` maps the same face
current used by `HJMagneticSimulation`'s `current_biot` receiver path: it
applies `average_face_to_cell_vector` and then the cell-current Biot integral.
`face_basis_biot_matrix(mesh, locations, subdivisions)` is the small-mesh
diagnostic matrix for face-basis reconstructed currents.  Tests verify that

```text
einsum("kcf,f->kc", B_face_current, q_face)
  == biot_savart_h_from_cell_currents(
       mesh,
       reshape(A_fc q_face, (n_cells, 3), order="F"),
       locations).

einsum("kcf,f->kc", B_face_basis, u_face)
  == biot_savart_h_from_face_basis_currents(mesh, u_face, locations).
```

This gives the next derivation a concrete finite-dimensional target: express
the source-history correction as a source-edge vector or a small set of local
edge-current basis vectors, then apply `B_edge` to obtain the receiver response.
The same pattern now covers the recovery paths that showed different empirical
coefficients, so the remaining work is to derive the source-history vector or
local edge basis that should be fed through these matrices.

The helper `source_history_receiver_basis` now constructs the simplest version
of that target explicitly:

```text
R_p(t_n, r_k) = g_p(t_n) B_R(r_k) s_0
```

where `B_R` is one of the explicit receiver matrices and `s_0` is the
mesh-projected initial grounded-source edge vector.  It returns a tensor with
shape

```text
(n_times, n_basis, n_locations, 3).
```

This is still an ansatz, not the production coefficient formula.  Its value is
that any proposed non-fitted coefficient vector `a_p` can now be tested as a
plain matrix contraction against the same receiver operator used by the runtime
magnetic recovery.

The diagnostic least-squares counterpart is `fit_source_history_coefficients`:

```text
target(t_n, r_k) ~= sum_p a_p R_p(t_n, r_k).
```

This is not a replacement for the derivation, but it gives every proposed
non-fitted coefficient formula the same acceptance interface.

The next local-operator scaffold is now explicit in
`atem3d.local_coupling`.  It builds canonical edge-current basis vectors over
the union of

```text
1. cells adjacent to nonzero grounded-source edges, and
2. cells nearest to magnetic receiver locations,
```

with optional integer cell-radius expansion.  The resulting
`LocalEdgeBasis.basis_vectors` are one-hot edge-current vectors on the local
support, and `source_history_receiver_basis_from_spatial_vectors` forms the
Cartesian product

```text
R_{p,s}(t_n, r_k) = g_p(t_n) B_R(r_k) v_s
```

for every Debye history order `p` and local spatial edge basis `v_s`.

This is still diagnostic infrastructure, not a production source-history
coefficient formula.  Its purpose is to let the next derivation step ask a
finite-dimensional question: can the missing source-primary/IP magnetic
recovery residual be represented by a compact local edge-current basis with
stable, mesh-transferable coefficients computed from FV/MMR operators rather
than fitted from empymod?

The single-report matrix CLI can audit this local space directly:

```text
atem3d-source-history-matrix-fit ip_report.json noip_report.json \
  --receiver-matrix current_biot \
  --spatial-basis local_edges \
  --local-basis-scope source_edges \
  --source-cell-radius 0 \
  --receiver-cell-radius 0 \
  --max-order 1 \
  -o local_edge_matrix_fit.json
```

The report records the local source/receiver support, expanded cell and edge
indices, flattened `g_p v_s` basis labels, and least-squares design diagnostics.
For local one-hot edge bases the CLI intentionally omits
`coefficients_over_mu_delta_l2`, because those coefficients live in the selected
receiver-matrix edge-current space rather than a global source-primary
normalization.  `--local-basis-scope` selects which local edge set becomes the
actual spatial dictionary:

```text
support_edges     = all edges touching source-adjacent and receiver-nearest cells
source_cell_edges = all edges touching source-adjacent cells
source_edges      = only nonzero projected grounded-wire source edges
source_moments    = low-order longitudinal moments xi^p s_0 on source_edges
```

The batch scan CLI accepts the same `--spatial-basis local_edges` option, so the
same local support can be audited over several BE orders and time windows.

Two first audits with this local basis are now saved:

```text
outputs/source_history_matrix_fit_current_biot_sub5_local_edges_hz_x0.json
  relative_l2 = 0.15697
  design_shape = [91, 688]
  rank = 2
  support = 67 cells / 344 edges

outputs/source_history_matrix_fit_current_biot_line_hz_local_edges_t010_1ms.json
  relative_l2 = 0.18445
  design_shape = [273, 720]
  rank = 6
  support = 69 cells / 360 edges

outputs/source_history_matrix_fit_current_biot_line_hz_source_edges_t010_1ms.json
  relative_l2 = 0.18445
  design_shape = [273, 44]
  rank = 6
  basis = 22 nonzero source edges

outputs/source_history_matrix_scan_current_biot_line_hz_local_edges.json
  support = 360 edges
  order 1 all:          relative_l2 = 0.18445, rank = 6
  order 1 0.10-0.30 ms: relative_l2 = 0.11328, rank = 6
  order 1 0.40-0.70 ms: relative_l2 = 0.00568, rank = 6
  order 1 0.80-1.00 ms: relative_l2 = 0.00036, rank = 6
  order 2 all:          relative_l2 = 0.10847, rank = 9
  order 2 0.10-0.30 ms: relative_l2 = 0.03132, rank = 9
  order 2 0.40-0.70 ms: relative_l2 = 0.00120, rank = 9
  order 2 0.80-1.00 ms: relative_l2 = 0.00004, rank = 9

outputs/source_history_matrix_scan_current_biot_line_hz_source_edges.json
  basis = 22 source edges
  same relative_l2 values and ranks as the support_edges scan
  design shapes reduce from [*, 720]/[*, 1080] to [*, 44]/[*, 66]

outputs/source_history_matrix_scan_current_biot_line_hz_source_moments_deg2.json
  basis = 3 source-line moment vectors: s_0, xi s_0, xi^2 s_0
  same relative_l2 values and ranks as the 22-edge source_edges scan
  design shapes reduce further to [*, 6]/[*, 9]

outputs/source_history_matrix_scan_current_biot_line_hz_source_moments_even02.json
  basis = 2 even source-line moment vectors: s_0, xi^2 s_0
  same relative_l2 values as the degree-2 full-moment scan
  design shapes reduce again to [*, 4]/[*, 6]

source_moments degree comparison:
  degree 0, order 1 all:          relative_l2 = 0.18741, rank = 2
  degree 1, order 1 all:          relative_l2 = 0.18741, rank = 4
  degree 2, order 1 all:          relative_l2 = 0.18445, rank = 6
  degree 0, order 2 0.10-0.30 ms: relative_l2 = 0.09425, rank = 3
  degree 1, order 2 0.10-0.30 ms: relative_l2 = 0.09425, rank = 6
  degree 2, order 2 0.10-0.30 ms: relative_l2 = 0.03132, rank = 9
```

The fit/scan JSON reports now include a `coefficient_table` that reshapes the
flat coefficients into `BE history x source moment`.  For the degree-2 scan:

```text
order 1 all:
                 source_moment:0   source_moment:1   source_moment:2
BE relaxation      -6.08e-05          2.98e-19          2.22e-05
BE cascade 1        1.88e-04         -6.61e-18         -6.74e-04

order 2, 0.10-0.30 ms:
                 source_moment:0   source_moment:1   source_moment:2
BE relaxation       3.13e-05          1.33e-16          8.76e-05
BE cascade 1       -1.46e-02         -1.90e-14         -1.08e-02
BE cascade 2        1.08e+00         -8.44e-15          7.15e-01
```

The single-fit CLI can also evaluate prescribed coefficients without refitting:

```text
atem3d-source-history-matrix-fit ip_report.json noip_report.json \
  --receiver-matrix current_biot \
  --spatial-basis local_edges \
  --local-basis-scope source_moments \
  --source-moment-degrees 0,2 \
  --max-order 1 \
  --prescribed-coefficients=-6.0778e-05,2.2238e-05,1.8810e-04,-6.7416e-04 \
  -o prescribed_order1.json
```

The leading `=` is important when the first coefficient is negative, otherwise
`argparse` can interpret it as another option.  The report
`outputs/source_history_matrix_fit_current_biot_line_hz_source_moments_even02_prescribed_order1.json`
confirms that the prescribed coefficients reproduce the fitted order-1 residual:

```text
fit relative_l2        = 0.1844477688670203
prescribed relative_l2 = 0.18444776886702025
```

This provides the acceptance hook for the next production step: once a
non-fitted FV/MMR formula proposes the `{g_p, xi^q s_0}` coefficients, it can be
passed through the prescribed path and compared against the optimal diagnostic
fit without using empymod to determine the coefficients.

The same prescribed-coefficient contract is now available in the EB runtime and
the H/J runtime via `magnetic_recovery_source_history`.  A single-term config
block

```yaml
magnetic_recovery_source_history:
  kind: prescribed_source_moments
  tau: 0.02
  max_order: 1
  source_moment_degrees: [0, 2]
  coefficients: [c00, c02, c10, c12]
  receiver_matrix: auto
```

or a multi-term block

```yaml
magnetic_recovery_source_history:
  terms:
    - tau: 0.0005
      max_order: 1
      source_moment_degrees: [0, 2]
      coefficients: [c00, c02, c10, c12]
    - tau: 0.002
      max_order: 1
      source_moment_degrees: [0, 2]
      coefficients: [d00, d02, d10, d12]
```

adds

```text
H_corr(t_n, r_k) =
  sum_m sum_{p,q} c_{m,p,q} g_{m,p}(t_n) B_R(r_k) [xi^q s_0]
```

to Biot/MMR magnetic receiver modes.  `receiver_matrix: auto` uses the same
receiver path as the selected magnetic recovery mode (`current_biot`,
`edge_current`, or `edge_basis` for EB; `face_current` or `face_basis` for H/J).
For H/J, the source moment vector is built from the signed face source used in
the H/J Ampere balance, `-source.initial_face_vector(mesh)`, so it is consistent
with `j = C h - s_e`.  This is an evaluation hook only: the coefficients must
come from a derivation or an explicitly documented diagnostic experiment, and
the block is disabled by default.  No empirical coefficient is hard-coded into
the runtime source-history path.

The runtime now also has a second disabled-by-default diagnostic kind:

```yaml
magnetic_recovery_source_history:
  kind: initial_polarization_source_moments
  source_moment_degrees: [0, 2]
  receiver_matrix: auto
  projection: receiver_l2
```

For each Debye term this builds the initial source-primary polarization current

```text
q_p^0 = -M_delta_sigma y^0
```

or its H/J face-current analogue, projects `q_p^0` onto the selected
source-moment vectors, and advances the projected field with the same BE
relaxation `g_0(t)` used by the memory variable.  The projection is computed
from finite-volume operators and the selected receiver matrix only; it does not
look at empymod residuals.  This makes the obvious first-principles candidate
testable in the same runtime path as the prescribed coefficients.

The source-centered H/J `current_biot` audit is saved in
`outputs/initial_polarization_source_history_candidate_source_centered_eval.json`.
It rejects this simple candidate as the production correction: against the
three `Hz` IP-residual columns from `0.10-1.00 ms`, the relative L2 is about
`41`, the candidate norm is `4.65e-4` versus target norm `1.14e-5`, and the
center receiver has the wrong sign.  Therefore the missing early-time MMR term
is not merely the initial Debye polarization current projected onto source
moments with a `g_0` decay.  The hook remains useful for ruling out such local
operator candidates without fitting coefficients.

A second non-fitted scale check used the material source-primary clue from the
empymod-only polarization scan: a coefficient proportional to
`mu0 * delta_sigma * L^2` and a slower `2*tau` relaxation.  The prescribed H/J
source-centered report
`outputs/diagnostic_hj_source_centered_material_scale_tau2_source_moment0.json`
tests the dimensionless coefficient `-2` on the degree-zero face-source moment.
It gives relative L2 `1.178` against the IP residual, while the least-squares
coefficient for that same single basis is `+10.03` in the same normalization
with relative L2 `0.340`.  This rejects the simple material-scale sign and
magnitude for the H/J `current_biot` residual.  Any future material-scaling law
must therefore be derived with the H/J receiver operator and the early-time
source convention included, not copied from the empymod-only source-primary
scale.

The same conclusion now holds through the H/J runtime path.  The diagnostic
flag `magnetic_recovery_source_primary_delta6` is available for H/J Biot
receiver modes with either the analytic `wire` basis or the FV `face_current`
source basis.  On
`outputs/validation_sweep_hj_source_centered_delta6_t010_1ms.json`, the
uncorrected source-centered `current_biot` `Hz` relative L2 values are
`0.588/0.650/0.589`; the delta6 wire basis worsens them to
`0.862/1.031/0.863`, and the face-current basis worsens them to
`0.889/1.035/0.890`.  Therefore the EB delta6 source-primary normalization is
not transferable to H/J, even when the source shape is evaluated with the H/J
face-current operator.

That face-current diagnostic is now also available through the source-history
runtime as `source_primary_delta6_source_moments`.  The new regression
`test_hj_source_primary_delta6_source_history_matches_face_current_path`
verifies that this path reproduces
`magnetic_recovery_source_primary_delta6_basis: face_current` at the receiver
data level.  Postprocessing
`outputs/hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_data_only.h5`
with
`examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta6_source_history.yaml`
and comparing against empymod gives the same H/J `Hz` relative L2 values,
`0.889/1.035/0.890`, in
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta6_source_history_t010_1ms_validation.json`.
This removes one more ambiguity: the H/J delta6 failure is not caused by the
source-history replay machinery or the degree-zero face-source moment.  The
missing production term is still the non-fitted kernel/amplitude/recovery law.

The matrix-fit and matrix-scan CLIs now expose the same H/J face-source space
through:

```text
--field-location face
--receiver-matrix current_biot|face_current|face_basis
--spatial-basis source_moments
```

with `--receiver-matrix current_biot` mapped to the H/J face-current Biot
matrix.  For `--spatial-basis source_moments` and `--spatial-basis local_edges`,
the JSON reports now also include a `static_response_matrix` block.  This block
is the rank audit of the selected scalar receiver columns of `B_R v_s` before
the backward-Euler time-history basis is multiplied in; it records the shape,
rank, singular values, column norms, and column-normalized conditioning.  A
rank-deficient `static_response_matrix` means the chosen receiver/component set
cannot distinguish all spatial source-history moment vectors even before any
time-window fitting.  Single-fit and batch-scan reports also include
`spatial_time_series`, which
projects each sampled target time gate onto the static source-moment/local-edge
space before any Debye time-basis fit is attempted.  This separates the spatial
question from the remaining source-history time-kernel question.

A first center-receiver audit on the sampled xy6 H/J reports is saved as

```text
outputs/source_history_matrix_scan_hj_face_current_reference_delta_hz_x0_source_moments_even02.json
```

using `target=reference_delta`, `Hz@x=0`, `tau=1 ms`, source face moments
`{s_0, xi^2 s_0}`, and `subdivisions=1`.  The all-window order-one fit gives
relative L2 `0.13643` with rank `2` for a four-column design; order two gives
relative L2 `0.07075` with rank `3` for a six-column design.  Windowed fits are
much smaller after the earliest gates (`0.00480` for `0.4-0.7 ms` at order one,
`0.00067` at order two; `0.000445` for `0.8-1.0 ms` at order one,
`0.000024` at order two), but they remain rank deficient.  The refreshed report
shows the static spatial cause directly:

```text
static_response_matrix shape = [1, 2]
static_response_matrix rank  = 1
```

For a single center `Hz` scalar column, the two even face-source moments are
therefore indistinguishable up to scale before the time basis is applied.  This
is useful constraint evidence for the H/J derivation: compact face-source
moments span a large part of the late-time center `Hz` source-history shape, but
the fitted coefficients are not unique and cannot be promoted to production
physics.

A second H/J audit uses the full three-point `Hz` line and fits the actual
Debye-IP residual (`target=ip_residual`) rather than the IP-minus-no-IP
reference delta:

```text
outputs/hj_debye_ip_line_exeyhz_xy6pad_data_only_t010_1ms_samples_report.json
outputs/source_history_matrix_scan_hj_face_current_ip_residual_hz_line_source_moments_even02.json
```

The three-point line removes the static spatial rank deficiency for the same
two even source moments:

```text
static_response_matrix shape = [3, 2]
static_response_matrix rank  = 2
```

The all-window order-one fit gives `relative_l2 = 0.05576` with rank `4`, and
order two gives `0.05006` with rank `6`.  Late windows again fit very well
(`0.00310` for `0.4-0.7 ms` at order one and `0.000774` at order two), but the
coefficients are still diagnostic because they are obtained from empymod
residuals.  The important new constraint is that the line receivers provide
enough static spatial rank for the even face-source moments, so future
non-fitted H/J source-history/MMR coefficients should be tested on the full
line rather than only the center `Hz` column.

The new `spatial_time_series` projection sharpens the diagnosis further.  For
the same three-point `Hz` IP residual, projecting each time gate onto the two
static face moments gives all-window `relative_l2 = 0.000656`, so the static
source-moment space is almost sufficient.  The remaining error is in the time
kernel: increasing the BE cascade order improves the all-window fit but with
rapidly worsening conditioning:

```text
outputs/source_history_matrix_scan_hj_face_current_ip_residual_hz_line_source_moments_even02_orders1to6.json
  order 1 all: relative_l2 = 0.05576, cond = 2.51e1
  order 2 all: relative_l2 = 0.05006, cond = 2.22e2
  order 3 all: relative_l2 = 0.03480, cond = 2.94e3
  order 4 all: relative_l2 = 0.02033, cond = 5.23e4
  order 5 all: relative_l2 = 0.01076, cond = 1.16e6
  order 6 all: relative_l2 = 0.00575, cond = 3.10e7
```

This argues against fixing H/J IP `Hz` with ever-higher fitted BE cascades.  The
production path still needs a non-fitted local FV/MMR source-history operator or
a better derived time kernel; the high-order fits are conditioning diagnostics,
not accepted physics.

The same audit has now been repeated with the H/J runtime itself configured to
use `magnetic_receiver_mode: current_biot`, so the residual and receiver matrix
refer to the same magnetic recovery path:

```text
examples/hj_debye_ip_line_exeyhz_xy6pad_current_biot.yaml
outputs/hj_debye_ip_line_exeyhz_xy6pad_current_biot_data_only_t010_1ms_samples_report.json
outputs/source_history_matrix_scan_hj_current_biot_ip_residual_hz_line_source_moments_even02_orders1to6.json
```

Relative L2 for `Hz` improves from the stored-H report but is still not an
accepted solution:

```text
current_biot H/J Debye, no source-history correction:
  Hz@x=-20 relative_l2 = 0.74307
  Hz@x=0   relative_l2 = 0.80856
  Hz@x=20  relative_l2 = 0.74164
```

The current-Biot residual has the same static spatial conclusion
(`static_response_matrix shape = [3, 2]`, rank `2`) and a small static
projection error (`spatial_time_series.projection_fit.relative_l2 = 0.000981`).
Its time-kernel scan is somewhat different from the stored-H residual:

```text
order 1 all: relative_l2 = 0.09587, cond = 2.51e1
order 2 all: relative_l2 = 0.02783, cond = 2.22e2
order 3 all: relative_l2 = 0.02052, cond = 2.94e3
order 4 all: relative_l2 = 0.01935, cond = 5.23e4
order 5 all: relative_l2 = 0.01531, cond = 1.16e6
order 6 all: relative_l2 = 0.01127, cond = 3.10e7
```

As a runtime-sign diagnostic only, the order-two fitted coefficients from this
same-path scan were inserted into
`examples/hj_debye_ip_line_exeyhz_xy6pad_current_biot_diagnostic_source_history_order2.yaml`.
The resulting report
`outputs/hj_debye_ip_line_exeyhz_xy6pad_current_biot_diagnostic_source_history_order2_t010_1ms_samples_report.json`
shows:

```text
current_biot H/J Debye with fitted diagnostic source-history:
  Hz@x=-20 relative_l2 = 0.02067
  Hz@x=0   relative_l2 = 0.02223
  Hz@x=20  relative_l2 = 0.02090
```

This verifies that the H/J runtime hook applies the face source-history moment
correction with the expected sign and scale.  It is not a production result:
the coefficients are fitted from empymod residuals, and the same run leaves the
two side `Ex` components at relative L2 about `0.740`, so source/receiver
normalization, primary-secondary treatment, or boundary/model convention errors
remain outside this magnetic MMR correction.

The batch scan CLI supports the same prescribed mode.  Using the all-window
optimal even-moment coefficients as fixed prescribed values gives:

```text
outputs/source_history_matrix_scan_current_biot_line_hz_source_moments_even02_order1all_prescribed.json
  order 1 all:          fit = 0.18445, prescribed = 0.18445
  order 1 0.10-0.30 ms: fit = 0.11328, prescribed = 0.34289
  order 1 0.40-0.70 ms: fit = 0.00568, prescribed = 0.11387
  order 1 0.80-1.00 ms: fit = 0.00036, prescribed = 0.13684

outputs/source_history_matrix_scan_current_biot_line_hz_source_moments_even02_order2all_prescribed.json
  order 2 all:          fit = 0.10847, prescribed = 0.10847
  order 2 0.10-0.30 ms: fit = 0.03132, prescribed = 0.20454
  order 2 0.40-0.70 ms: fit = 0.00120, prescribed = 0.05216
  order 2 0.80-1.00 ms: fit = 0.00004, prescribed = 0.07778

outputs/source_history_matrix_scan_current_biot_line_hz_source_moments_even02_order123.json
  order 3 all:          relative_l2 = 0.05689, cond = 1.98e7
  order 3 0.10-0.30 ms: relative_l2 = 0.00845, cond = 1.63e9
  order 3 0.40-0.70 ms: relative_l2 = 0.00018, cond = 5.03e8
  order 3 0.80-1.00 ms: relative_l2 = 0.000004, cond = 1.63e9

outputs/source_history_matrix_scan_current_biot_line_hz_source_moments_even02_order3all_prescribed.json
  order 3 all:          fit = 0.05689, prescribed = 0.05689
  order 3 0.10-0.30 ms: fit = 0.00845, prescribed = 0.10877
  order 3 0.40-0.70 ms: fit = 0.00018, prescribed = 0.03355
  order 3 0.80-1.00 ms: fit = 0.000004, prescribed = 0.04682
```

This rejects another tempting shortcut: fixed all-window empirical coefficients
do not transfer to subwindows, even in the compact even-moment basis.  Adding a
third same-pole BE cascade improves the optimal in-window residual, but it also
makes the design matrix severely ill-conditioned and still fails the fixed-
coefficient transfer test.  A production correction still has to come from the
discrete source-history/FV-MMR operator, not from freezing least-squares
coefficients or growing the empirical history basis.

This is useful structural evidence.  The large support dictionary gives exactly
the same residuals as the 22-edge source-only dictionary, so the receiver-cell
edges are redundant for the current `Hz` source-history residual.  The missing
term is therefore source-support dominated in this audit.  More importantly,
the 22 one-hot source edges collapse to only three longitudinal source-line
moments without losing fit quality.  The odd first moment does not help this
symmetric `Hz` geometry by itself; the quadratic even moment is what recovers
the source-edge dictionary space.  Removing the odd moment entirely keeps the
same residuals, leaving the minimal useful spatial audit space
`{s_0, xi^2 s_0}` for this symmetric receiver line.  Adding BE order 2 reduces
residuals, especially after the earliest gates, but increases the time-history
rank from 4 to 6 in the even-moment space.  The production operator should now
be derived on this low-dimensional even source-line moment space, not on a
larger empirical edge dictionary.

`a_p` should be checked by applying it to `R_p`; the fitted `a_p` is only an
audit value showing what the local matrix basis would need to reproduce a
reference residual.

For sampled validation reports that contain scalar columns such as `Hz@x=0`,
`fit_source_history_coefficients_for_components` performs the same fit after
selecting `(receiver_index, component_index)` pairs from the response tensor.
This keeps the matrix-basis audit compatible with the existing empymod
validation reports, which often store only a subset of `Ex/Ey/Hz` components.
The CLI `atem3d-source-history-matrix-fit` now exposes this for
`--receiver-matrix current_biot`, `edge_current`, and `edge_basis`.  On the
strict one-Debye three-point `current_biot` residual, the `current_biot` matrix
basis gives normalized BE coefficients about `[-5.834, -2.013]`, while the
edge-current source basis gives `[-6.340, -2.176]`; this confirms that the
matrix interface is sensitive to the chosen recovery operator and can audit
future non-fitted coefficient formulas against that operator.

The same CLI also exposes source-vector candidates through `--source-vector`:

```text
wire                    = s_0
dc_conduction_current   = M_sigma0 e_0
dc_total_current        = M_sigma0 e_0 + s_0
dc_polarization_current = sum_i M_delta_i e_0
```

On the same strict `current_biot` residual with the `current_biot` receiver
matrix, the first candidate scan gives:

```text
wire:                    [-5.834, -2.013], relative_l2 = 0.0882
dc_total_current:        [-5.873, -2.021], relative_l2 = 0.0844
dc_conduction_current:   [433.735, 231.439], relative_l2 = 0.7441
dc_polarization_current: [2168.674, 1157.193], relative_l2 = 0.7441
```

This does not derive the production coefficients, but it rules out the isolated
DC conduction and initial polarization currents as the dominant source-shaped
spatial basis on this diagnostic.  The useful basis remains tied to the
grounded-source total current system, with `dc_total_current` slightly better
than the wire-only vector on this mesh/report.

A follow-up scan on available `current_biot` reports shows that this slight
`dc_total_current` advantage is not stable enough to promote into the production
formula:

```text
sub1 three-point wire:       [-5.834, -2.013], relative_l2 = 0.0882
sub1 three-point dc_total:   [-5.873, -2.021], relative_l2 = 0.0844
sub5 three-point wire:       [-10.132,  2.752], relative_l2 = 0.0435
sub5 three-point dc_total:   [-10.187,  2.766], relative_l2 = 0.0444
xy6 sub5 center wire:        [-10.041,  2.610], relative_l2 = 0.0409
xy6 sub5 center dc_total:    [-10.029,  2.607], relative_l2 = 0.0409
xy8 sub5 center wire:        [-9.462,   1.424], relative_l2 = 0.0326
xy8 sub5 center dc_total:    [-9.451,   1.422], relative_l2 = 0.0326
```

Thus the next derivation should not spend effort distinguishing `s_0` from
`M_sigma0 e_0 + s_0` as the primary unresolved issue.  The dominant unresolved
object remains the recovery-path/window-dependent coefficient coupling that
moves the fitted BE coefficients from roughly `[-5.8, -2.0]` in the sub1 audit
to roughly `[-10, +2.6]` in the sub5/window audits.

The matrix-fit CLI now also supports `--time-min` and `--time-max` to fit the
same report over restricted sampled time windows.  On the sub1 three-point
`current_biot` residual with the wire source vector and order-one BE basis:

```text
0.10-1.00 ms: [-5.834, -2.013], relative_l2 = 0.0882
0.10-0.30 ms: [-4.844, -7.035], relative_l2 = 0.1280
0.40-0.70 ms: [-6.051, -1.772], relative_l2 = 0.0375
0.80-1.00 ms: [-6.918, -0.533], relative_l2 = 0.0240
```

Adding one more same-pole cascade (`max_order=2`) gives only a small all-window
improvement:

```text
order 1 all:  relative_l2 = 0.0882, coeff = [-5.834, -2.013]
order 2 all:  relative_l2 = 0.0843, coeff = [-5.291, -5.053, 6.144]
order 2 early relative_l2 = 0.1124, coeff = [-0.704, -55.067, 240.044]
```

The matrix-fit reports now include design-matrix diagnostics.  On the same
`current_biot`/wire basis scan, the all-window order-one fit has condition
number `4.91` (`3.72` after column normalization), while the early-window
order-one fit already rises to `17.36` (`6.55` column-normalized).  The
order-two all-window fit has condition number `43.44` (`18.00`
column-normalized), and the order-two early-window fit rises to `650.81`
(`53.22` column-normalized).  The early-window order-two coefficients are
therefore large and cancellation-dominated, so the remaining error is unlikely
to be fixed by simply adding more empirical same-pole basis terms.  The
production path should instead derive the local early-time MMR/source coupling
or replace this scalar source-history ansatz with a finite-volume local receiver
solve whose convergence can be verified.

The scan CLI `atem3d-source-history-matrix-scan` now runs this audit over
multiple orders and windows in one report.  On
`outputs/source_history_matrix_scan_current_biot_sub1_wire.json`, the same
sub1 `current_biot`/wire residual gives:

```text
order 1 all:       relative_l2 = 0.08817, cond = 4.91
order 1 0.10-0.30: relative_l2 = 0.12802, cond = 17.36
order 1 0.40-0.70: relative_l2 = 0.03749, cond = 14.61
order 1 0.80-1.00: relative_l2 = 0.02397, cond = 29.87
order 2 all:       relative_l2 = 0.08430, cond = 43.44
order 2 0.10-0.30: relative_l2 = 0.11241, cond = 650.81
order 2 0.40-0.70: relative_l2 = 0.03742, cond = 376.98
order 2 0.80-1.00: relative_l2 = 0.02397, cond = 1219.95
```

The later windows have very small error reductions when `g_2` is added, but
their condition numbers are even worse.  This batch result reinforces that a
derived local operator should be judged by transfer and conditioning, not only
by least-squares residual.

The matrix-fit CLI also supports `--per-column`, which fits independent
coefficients for each sampled receiver/component column while keeping the same
receiver matrix and BE basis.  On
`outputs/source_history_matrix_fit_current_biot_sub1_wire_per_column.json`, the
three-point `current_biot`/wire `Hz` residual gives:

```text
common:  [-5.834, -2.013], relative_l2 = 0.0882
x=-20 m: [-5.203, -2.672], relative_l2 = 0.0761
x=0 m:   [-6.619, -1.192], relative_l2 = 0.0368
x=20 m:  [-5.203, -2.672], relative_l2 = 0.0761
```

The side receivers remain symmetric, but the center receiver needs a different
coefficient pair.  This is another constraint on the production derivation: the
missing source-history/MMR term cannot be only a global material-time kernel.
It must allow geometry- or local-recovery-dependent coupling while still being
computed from finite-volume operators rather than fitted from empymod.

The scan CLI can combine `--per-column` with order/window scans.  The report
`outputs/source_history_matrix_scan_current_biot_sub1_wire_per_column.json`
records `receiver_location` for each fitted column and shows that the geometry
dependence is stable in symmetry but not constant in time-window:

```text
order 1 all:
  x=-20 m [-5.203, -2.672], relative_l2 = 0.0761
  x=0 m   [-6.619, -1.192], relative_l2 = 0.0368
  x=20 m  [-5.203, -2.672], relative_l2 = 0.0761

order 1 0.10-0.30 ms:
  x=-20 m [-3.775, -9.831], relative_l2 = 0.1006
  x=0 m   [-6.177, -3.552], relative_l2 = 0.0555
  x=20 m  [-3.775, -9.831], relative_l2 = 0.1006

order 1 0.40-0.70 ms:
  x=-20 m [-5.642, -2.105], relative_l2 = 0.00293
  x=0 m   [-6.560, -1.356], relative_l2 = 0.00154
  x=20 m  [-5.642, -2.105], relative_l2 = 0.00293

order 1 0.80-1.00 ms:
  x=-20 m [-6.666, -0.639], relative_l2 = 0.00058
  x=0 m   [-7.232, -0.401], relative_l2 = 0.00039
  x=20 m  [-6.666, -0.639], relative_l2 = 0.00058
```

The late-window per-column residuals are nearly zero, so the selected basis can
describe local columns after fitting.  The coefficient drift across windows and
receiver positions is the unresolved production problem: the next formula must
compute these local couplings from the FV/MMR source-recovery operator instead
of fitting each column.

The helper `source_history_receiver_basis_from_vectors` and CLI
`--source-vectors name0,name1,...` now allow each BE history order to use a
different static FV source vector.  This checks whether the instability above
comes from forcing `g_0` and `g_1` to share the same spatial source shape.  On
the sub1 `current_biot` residual:

```text
same wire, all:              relative_l2 = 0.0882
wire, dc_total, all:         relative_l2 = 0.0878
dc_total, wire, all:         relative_l2 = 0.0848
wire, dc_conduction, all:    relative_l2 = 0.1139
wire, dc_polarization, all:  relative_l2 = 0.1139

same wire, 0.10-0.30 ms:      relative_l2 = 0.1280
wire, dc_total, 0.10-0.30 ms: relative_l2 = 0.1272
dc_total, wire, 0.10-0.30 ms: relative_l2 = 0.1246
```

Thus allowing separate static source vectors by BE order gives only a marginal
improvement and does not remove the early-time residual.  The isolated DC
conduction and polarization-current shapes remain poor candidates.  This rules
out another simple scalar-basis extension and reinforces that the missing term
is a local MMR/source-coupling operator, not just a better choice of static
source vector for the existing `g_p(t) B_R s_p` ansatz.

## 11. H/J Magnetic-Diffusion Recovery Spectrum

The rejected static-current candidates above all try to choose a better vector
inside the existing scalar history form.  The evidence now points to a local
recovery operator instead.  For the no-source H/J magnetic equation,

```text
M_mu dh/dt + C.T M_rho C h = 0,
```

the homogeneous recovery modes satisfy the generalized eigenproblem

```text
C.T M_rho C h_k = lambda_k M_mu h_k,
tau_k = 1 / lambda_k.
```

The positive `lambda_k` values are finite-volume recovery rates for the same
curl-curl operator used by the H/J time step.  They are therefore a better
non-fitted source of local time scales than the empirical `fast_tau` scans:
the Debye source memory should drive one or more local magnetic recovery modes,
and the MMR/source-receiver matrix should then project those modes to `Hz`.

The helper module `atem3d.recovery_spectrum` now exposes:

```text
magnetic_diffusion_matrices(mesh, conductivity, mu)
magnetic_diffusion_positive_spectrum(mesh, conductivity, max_modes)
magnetic_diffusion_time_constants(mesh, conductivity, max_modes)
```

This implementation is intentionally dense and limited to small local
supports, because the production problem is not the full-domain spectrum.  The
next production step is to define the local support and forcing/projection
operator, then verify that the resulting modal source-history correction
reproduces the saved `delta_residual` traces without fitting coefficients to
empymod.  Until that forcing/projection map is derived, the spectrum is a
diagnostic building block rather than the accepted near-source `Hz` correction.

The runtime hook can now carry this modal shape without changing the H/J solve.
`driven_recovery_source_moments` accepts either a scalar `response_tau` or a
list `response_taus`.  With multiple response times, the supplied coefficient
array is interpreted as

```text
coefficients[mode_index, source_moment_index],
```

and the receiver correction is

```text
Delta H(t) = sum_k z_k(t) B_R S c_k,
z_k^{n+1} = a_k z_k^n + b_k g_driver^{n+1},
a_k = tau_k / (tau_k + dt_n),
b_k = dt_n / (tau_k + dt_n).
```

This makes the next audit straightforward: compute `tau_k` from the local
magnetic-diffusion spectrum, derive or approximate the modal forcing/projection
coefficients from FV/MMR operators, and pass those non-fitted values through
the existing runtime and postprocess validation path.

The CLI `atem3d.recovery_spectrum_cli` records the local support and spectrum
as JSON.  It accepts explicit global cell indices or a source/receiver-derived
support.  For H/J source-centered runs, `--field-location face` uses the active
grounded-wire face-source dofs and the receiver-nearest cells:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.recovery_spectrum_cli `
  examples\hj_debye_ip_line_exeyhz_source_centered_current_biot.yaml `
  --support source_receiver `
  --field-location face `
  --source-cell-radius 1 `
  --receiver-cell-radius 1 `
  --padding 0 `
  --max-modes 6 `
  --conductivity-model sigma0 `
  -o outputs\local_recovery_spectrum_source_receiver_demo.json
```

The generated source-centered report uses 210 local cells and 906 H/J edge
dofs; the first six homogeneous local recovery time constants are `O(2e-8 s)`,
far faster than the empirical `O(0.1 ms)` rise-decay clue.  This means the
simple homogeneous local support spectrum is not, by itself, the missing
source-history law.  The next implementation step is to derive the modal
forcing/projection coefficients and audit whether a larger or differently
weighted source-receiver recovery support produces the observed slower
effective kernel before passing non-fitted `response_taus`/coefficients to
`driven_recovery_source_moments`.

The modal forcing/projection diagnostic is now available with
`--include-modal-coupling`.  It uses the discrete H/J source location

```text
f_s = C.T M_rho s_f
```

and the `M_mu`-orthonormal modes from

```text
C.T M_rho C h_k = lambda_k M_mu h_k
```

so that each source moment has modal drive

```text
a_{k,s} = h_k.T f_s.
```

For `face_current_biot` receiver projection, the receiver vector for mode `k`
is computed from `C h_k`; the saved `source_receiver_response` stores
`a_{k,s} / lambda_k` times that receiver vector.  On
`outputs\local_recovery_spectrum_source_receiver_modal_coupling_demo.json`, the
largest first-six-mode steady source/receiver factor is about `1.12e-2`, with
the source forcing concentrated mostly in two modes.  Because the corresponding
time constants are still `O(2e-8 s)`, this diagnostic rules out the simplest
small-support homogeneous modal explanation for the observed `O(0.1 ms)`
source-history build-up.  The spectrum filter records 359 discarded near-null
modes; the last discarded eigenvalue is `O(1e-3-1e-2 1/s)` while the first kept
mode jumps to about `4.8e7 1/s`.  The missing `lambda ~= 1e4 1/s` scale is
absent from this local homogeneous spectrum, so the production derivation must
look beyond this plain source/receiver support eigenproblem.

The same report now carries the algebraic bridge needed by the runtime
source-history hook: each modal `source_receiver_response[k, s]` is projected
back onto the static source-moment response matrix.  If

```text
A_s = B_R s_f^{(s)}
T_{k,s} = (a_{k,s}/lambda_k) B_R C h_k,
```

then the report solves

```text
min_c || A c_{k,s} - T_{k,s} ||_2
```

without using empymod residuals.  For the six-mode source-centered support,
this modal-to-source projection has aggregate receiver-space relative L2
`0.342`; individual mode/source-drive projection errors range from about
`0.234` to `1.0`.  Collapsing those projected coefficients with the compact
normalized amplitude `[8.1876, 2.0719]` gives the trace-kernel audit
`outputs/source_history_hj_rise_decay_kernel_prescribed_modal_projection_geometry_amp.json`.
It has prescribed trace relative L2 `0.3936`, per-trace L2
`0.3526/0.7216`, and summed normalized coefficients `[5.83, 4.01]`.
The same coefficients were put through the runtime/postprocess path in
`examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_modal_projection_geometry_amp.yaml`.
The empymod comparison
`outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_modal_projection_geometry_amp_t010_1ms_validation.json`
gives `Hz` relative L2 `0.295/0.350/0.296` at `x=-20/0/20 m`.  This is a real
non-fitted improvement over the uncorrected `0.588/0.650/0.589`, but it remains
worse than the trace-fitted geometry multi-tau diagnostic (`0.171/0.176/0.172`).
Therefore the current plain homogeneous local modal projection is rejected as
the production law.  The useful survivor is the bridge itself: a larger,
return-current-aware, or nonhomogeneous local recovery operator can now be
projected into the same runtime coefficient coordinates without fitting the
receiver residual.

The first nonhomogeneous local check is now in the same CLI.  With
`--include-driven-response`, the local support solves

```text
M_mu (h^{n+1} - h^n)/dt_n + C.T M_rho C h^{n+1}
  = C.T M_rho s_f g^{n+1},    h^0 = 0,
```

where `g` is the saved backward-Euler Debye driver.  The receiver trace
`B_R C h^n` is then projected to the source-moment coordinates by the same
least-squares map used for modal responses.  The source-centered report

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_demo.json
```

uses the `1 ms` driver and the same 210-cell source/receiver support.  When its
time-dependent source-drive projection is collapsed with the compact normalized
amplitude `[8.1876, 2.0719]`, the resulting normalized coefficient trace has
relative L2 `0.2168` against the saved H/J `delta_residual` trace.  Its optimal
global scale is `1.0067`, so the amplitude scale is close; the main mismatch is
early-time shape.  The candidate starts at `[7.40, 1.94]` at `0.1 ms` while the
target starts at `[0.820, 0.153]`, and both end near `[3.1, 0.79]` at `1 ms`.
This is better than the static homogeneous modal projection, but it still does
not close the production law.  The next operator must modify the early-time
recovery path, likely through return-current/global support or a constrained
nonhomogeneous solve rather than through another scalar amplitude.

The support-size check uses the driven-only `--skip-spectrum` path so larger
supports can be tested without a dense eigensolve.  Four source/receiver
supports were compared after collapsing with `[8.1876, 2.0719]`:

```text
support          cells  edges  trace L2  first coefficient pair
sr0 rr0 pad0       36    226   0.2177    [7.41, 1.88]
sr1 rr1 pad0      210    906   0.2168    [7.40, 1.94]
sr0 rr1 pad1      420   1653   0.2157    [7.38, 2.04]
sr1 rr1 pad1      560   2162   0.2157    [7.38, 2.04]
target                         --        [0.820, 0.153]
```

Thus the early overestimate is not cured by this local support enlargement.
Changing the driver from Debye decay `g(t)` to the monotone build-up
`1 - g(t)` has the opposite failure mode:

```text
support          trace L2  first pair       last pair
sr0 rr0 pad0     0.5683    [0.775, 0.197]  [5.16, 1.31]
sr1 rr1 pad1     0.5678    [0.772, 0.213]  [5.14, 1.42]
target           --        [0.820, 0.153]  [3.15, 0.789]
```

The first gate becomes physically plausible, but late time is too large.  The
missing source-history/MMR kernel is therefore neither a pure Debye decay nor a
pure monotone Debye build-up.  It needs a zero-initial rise-decay response, as
the earlier `g_slow - g_fast` trace-kernel evidence suggested, but with the fast
time and coefficients generated by a constrained/local-global recovery operator
rather than by fitting empymod residuals.

A simple charge-conserving projection of the face source before the local
driven solve has now been checked and ruled out.  The CLI option
`--driven-source-projection charge_conserving` replaces each source vector
`s_f` by

```text
s_f - M_rho^{-1} G phi,
```

where `phi` enforces discrete face-current continuity.  But the driven magnetic
forcing is

```text
C.T M_rho (s_f - M_rho^{-1} G phi)
  = C.T M_rho s_f - C.T G phi
  = C.T M_rho s_f.
```

The real report
`outputs/local_recovery_spectrum_source_receiver_driven_response_charge_conserving_demo.json`
confirms this invariance: compared with the raw-source driven report, the
maximum receiver-response difference is about `1.8e-13` and the maximum
projected-coefficient difference is about `2.1e-11`.  Thus endpoint
charge-conserving projection alone cannot fix the early-time overestimate; the
needed return-current effect has to enter through the support/operator itself,
not through a gradient current removed before `C.T M_rho`.

The next nonhomogeneous check lets the local driven solve start from a nonzero
MMR state.  The new CLI options

```text
--driven-initial-state charge_conserving_mmr
--driven-initial-state global_charge_conserving_mmr
```

either solve the MMR vector-potential problem on the local support, or solve it
on the full mesh and then restrict the global edge `h` state to the local
support.  Both modes first project each source-moment face current to a
charge-conserving current and then use the resulting nonzero edge state in the
same driven BE update.  This required the local support to expose
`global_face_indices` and `global_edge_indices`, so future audits can also
restrict full-domain H/J states to the local operator without guessing dof
ordering.

For the source-centered 210-cell support, the diagnostic report is

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_charge_conserving_mmr_initial.json
outputs/local_recovery_spectrum_source_receiver_driven_response_global_charge_conserving_mmr_initial.json
outputs/local_recovery_spectrum_charge_conserving_mmr_initial_trace_audit.json
```

On the sampled `0.1-1 ms` target window, both nonzero initial states give the
same trace result as the zero-initial driven response: fitted constant
source-drive amplitudes are `[8.191, 2.282]` in `mu*delta_sigma*L^2` units, with
trace L2 `0.2154`; using the compact `[8.1876, 2.0719]` amplitude gives trace
L2 `0.2168`.  The MMR initial state mainly affects `t=0` and then decays on the
very fast local magnetic diffusion scale.  Therefore the missing term is not
obtained by simply adding either a local or globally restricted
charge-conserving MMR initial state to the driven response; any full-domain
return-current effect has to enter as an ongoing boundary/operator coupling.

The first ongoing-forcing check is

```text
--driven-forcing global_mmr_steady
```

which replaces the usual local edge RHS `C.T M_rho s` by `K_local h_global`.
Here `h_global` is the full-mesh charge-conserving MMR source-moment state
restricted to the local support, and `K_local = C.T M_rho C`.  The source-
centered diagnostic files are

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_global_mmr_steady_forcing.json
outputs/local_recovery_spectrum_global_mmr_steady_forcing_trace_audit.json
```

This steady global MMR forcing does not improve the trace: the fitted-amplitude
relative L2 is still `0.2154`, the compact `[8.1876, 2.0719]` amplitude gives
`0.2183`, and the first selected fitted gate remains `[7.40, 2.13]` while the
target is `[0.82, 0.15]`.  Therefore the missing operator is not just the steady
global MMR state injected as a continuous RHS.  It must change the time kernel
itself, or supply a genuinely dynamic constrained boundary/operator coupling.

The driven-only path was then run with `driver_kind = relaxation_difference`.
This uses

```text
g_driver(t) = g_slow(t) - g_fast(t),
```

where `g_slow` is the `1 ms` Debye memory and, unless supplied explicitly,
`g_fast` uses the support estimate `tau_fast = mu0 * mean(sigma) * L_support^2`.
After collapsing the projected time-dependent source-moment matrix with the
compact normalized amplitude `[8.1876, 2.0719]`, the current reports give:

```text
support          cells  edges  fast_tau(s)        trace L2  opt scale  scaled L2
sr0 rr0 pad0       36    226   4.5238934e-5       0.1862    1.0327     0.1835
sr1 rr1 pad0      210    906   5.6330879e-5       0.1717    1.0489     0.1654
sr1 rr1 pad1      560   2162   2.0032567e-4       0.3209    1.3524     0.1940
```

The best non-fitted time-shape candidate so far is therefore the
`sr1/rr1/pad0` relaxation-difference driven solve:

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_sr1_rr1_pad0.json
```

It slightly improves on the prescribed single geometry-tau trace check
(`0.1731`) and approaches the fitted geometry multi-tau runtime diagnostic
(`0.171/0.176/0.172` in receiver validation).  The improvement is not a
production closure, because the amplitude `[8.1876, 2.0719]` still comes from
coefficient-sum evidence rather than from a derived FV/MMR source-recovery
operator.  The remaining derivation is now sharper: derive that compact
source-moment amplitude, or derive the full support-dependent matrix that
distributes it across recovery paths, without fitting empymod residual traces.

The same rise-decay driver was then combined with `global_mmr_steady` forcing:

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_global_mmr_steady_forcing_sr1_rr1_pad0.json
outputs/local_recovery_spectrum_relaxdiff_global_mmr_steady_forcing_trace_audit.json
```

This combination is not a breakthrough.  With a fitted two-component constant
amplitude, ordinary `source_edge_rhs` and `global_mmr_steady` both give trace L2
about `0.1637`; with the compact `[8.1876, 2.0719]` amplitude, the global-MMR
steady forcing is slightly worse (`0.1736`) than the ordinary source RHS
(`0.1717`).  The first fitted gate remains about `[6.05, 1.74]`, still far above
the target `[0.82, 0.15]`.  Thus replacing the steady spatial RHS does not
generate the required early-time suppression; the missing source-history law is
still a dynamic kernel/amplitude problem.

The best trace-space relaxation-difference clue was also pushed through the
existing runtime/postprocess hook as a single `driven_recovery_source_moments`
term:

```text
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp.yaml
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_scaled.yaml
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_t010_1ms_validation.json
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_scaled_t010_1ms_validation.json
```

The unscaled compact amplitude gives `Hz` relative L2
`0.2356/0.2497/0.2364`; multiplying by the trace-space optimal scalar `1.0489`
gives `0.2224/0.2364/0.2231`.  This is a real improvement over the uncorrected
H/J `current_biot` result (`0.588/0.650/0.589`) but worse than the earlier
one-mode driven diagnostic (`0.194/0.205/0.195`).  Therefore the
relaxation-difference local driven report should be read as a time-dependent
operator clue, not as a static single-mode runtime coefficient law.  A future
production path either needs to evaluate the local driven-response matrix
itself, or derive a reduced set of static modes that reproduces that matrix
without fitting the empymod residual.

The same static runtime hook was also replayed with the two-component
constant amplitude fitted directly from the sr1/rr1/pad0
`relaxation_difference` trace audit:

```text
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_fitted.yaml
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_fitted_data_only.h5
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_fitted_t010_1ms_validation.json
```

The normalized fitted amplitude is `[8.5343, 2.3771]`, corresponding to
physical coefficients `[5.3623e-5, 1.4936e-5]`.  Its runtime `Hz` relative L2
is `0.2208/0.2372/0.2216`.  This is only a small edge-receiver improvement over
the compact scaled replay and is slightly worse at the center receiver, so
replacing the compact amplitude by the best two-component trace amplitude does
not close the runtime gap.

The runtime hook now has a stricter diagnostic replay path for that question:

```yaml
magnetic_recovery_source_history:
  kind: time_series_source_moments
  times: [...]
  source_moment_degrees: [0, 2]
  coefficients: [[a_0(t_0), a_2(t_0)], ...]
```

At each saved time node it evaluates

```text
H_sh(t_n, x_r) = sum_s a_s(t_n) B_R v_s(x_r),
```

using the same FV/MMR receiver matrix `B_R` and source-moment vectors `v_s` as
the other hooks.  The sr1/rr1/pad0 relaxation-difference driven-response report
was collapsed with the compact physical amplitude and replayed through:

```text
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_time_series_geometry_amp.yaml
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_time_series_geometry_amp_scaled.yaml
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_time_series_geometry_amp_t010_1ms_validation.json
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_time_series_geometry_amp_scaled_t010_1ms_validation.json
```

The unscaled replay gives `Hz` relative L2 `0.2512/0.2667/0.2519`; the
trace-space scalar `1.0489` gives `0.2378/0.2528/0.2385`.  This is weaker than
the static scaled replay above.  Therefore even the explicit time-dependent
source-moment trace extracted from this local support is not yet a transferable
runtime correction.  The missing operation is not just "use the local projected
trace"; it must include the correct finite-volume source/receiver recovery map
that turns the local driven state into receiver-space `Hz`.

To rule out a projection/replay artifact, the same collapsed local
`receiver_response` tensor was applied directly to the three saved `Hz` data
columns, bypassing the source-moment table:

```text
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_receiver_response_geometry_amp_t010_1ms_validation.json
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_receiver_response_geometry_amp_scaled_t010_1ms_validation.json
```

The unscaled and scaled receiver-space results are identical to the
`time_series_source_moments` replay to the printed precision:
`0.2512/0.2667/0.2519` and `0.2378/0.2528/0.2385`.  Therefore the remaining
error is not caused by the static FV/MMR source-moment response matrix.
A negative-amplitude control,

```text
outputs/hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_receiver_response_geometry_amp_scaled_negative_t010_1ms_validation.json
```

worsens `Hz` to `0.973/1.097/0.974`, so the correction sign is also not the
problem.  The local magnetic-diffusion driven solve currently has a useful time
scale, but its forcing/support/amplitude is still not the missing grounded-wire
MMR source-recovery operator.

To separate local-support truncation from the source-history law, the
`source_receiver` support was padded to the full simulation mesh
(`21344` cells, `68778` edge dofs) and the driven solve was rerun with
`--skip-spectrum`.  The trace audits in this section are generated with
`atem3d.recovery_driven_trace_audit_cli`, which compares the
`recovery_spectrum` driven-response source-moment projection against the clean
`spatial_time_series` target.  With the geometry-derived fast time
`5.633087883e-5 s`, the full-domain dynamic report

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_fullpad_global_dynamic.json
outputs/local_recovery_spectrum_relaxdiff_fullpad_global_dynamic_trace_audit.json
```

gives fitted trace L2 `0.1628`, only marginally different from the local
sr1/rr1/pad0 value `0.1637`, while the compact amplitude worsens to `0.1903`
(`0.1853` after an optimal scalar).  The first fitted gate is still
`[6.03, 1.74]` in `mu*delta_sigma*L^2` units, far above the target
`[0.82, 0.15]`.

Repeating the full-domain dynamic solve with the empirical all-window
`fast_tau=0.11 ms` produces

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_fullpad_empirical_tau.json
outputs/local_recovery_spectrum_relaxdiff_fullpad_empirical_tau_trace_audit.json
```

and fitted trace L2 `0.1215`, essentially matching the earlier single
rise-decay trace fit.  However its first fitted gate remains
`[4.51, 1.30]`, still much larger than `[0.82, 0.15]`, and the compact
amplitude needs an optimal scalar `1.144` to reach L2 `0.1510`.  Thus a
full-domain dynamic magnetic-diffusion solve can reproduce the same empirical
all-window fast-time clue, but it does not by itself generate the early-time
suppression or non-fitted amplitude law.  The remaining term is still a
source-history/MMR kernel and amplitude derivation, not merely a larger support
or steady/global forcing replacement.

A source/receiver support sweep is saved in
`outputs\local_recovery_spectrum_source_receiver_sweep.json`.  It varies
source cell radius, receiver cell radius, and padding:

```text
sr rr pad   cells  edges  tau_est(s)   first kept tau(s)
0  0  0       36    226   4.52e-5     2.94e-7
0  0  1      210    906   5.63e-5     2.08e-8
0  1  0      144    643   5.63e-5     2.07e-8
0  1  1      420   1653   2.00e-4     4.48e-8
1  0  0      168    745   5.63e-5     2.07e-8
1  0  1      480   1879   2.00e-4     4.49e-8
1  1  0      210    906   5.63e-5     2.08e-8
1  1  1      560   2162   2.00e-4     skipped by dense_dof_limit
```

The geometric diffusion estimate reaches the right order of magnitude, but the
computed homogeneous spectra do not contain the corresponding eigenvalue band.
This narrows the remaining derivation: the `O(0.1 ms)` behavior is likely tied
to the source/receiver recovery path, outer-domain return currents, or a
nonhomogeneous driven solve, rather than to the plain local homogeneous
curl-curl spectrum.

The trace-kernel bridge
`outputs\source_history_hj_rise_decay_kernel_from_recovery_sweep_tau_estimates.json`
uses those sweep `tau_est` values as fast-time candidates in the existing
rise-decay coefficient-trace fit.  After duplicate tau values are removed, the
unique candidates are approximately:

```text
4.52e-5 s, 5.63e-5 s, 2.00e-4 s
```

The best single geometry-derived tau is `5.63e-5 s`, with trace relative L2
`0.164`.  This is worse than the empirical all-window best `0.11 ms` result
(`0.122`), but a three-column fit using all unique geometry-tau candidates
reaches relative L2 about `0.039` with column-normalized condition number about
`251`.  Thus support geometry contains the right family of time scales, but the
production formula still needs a non-fitted rule for the modal/source
coefficients and for which driven paths to include.

The report also records coefficient sums over the history basis.  For the
three geometry-tau columns, the physical sum is

```text
[5.1444e-5, 1.3018e-5]
```

and the normalized sum in units of `mu * delta_sigma * L^2` is

```text
[8.1876, 2.0719].
```

This is close to the earlier single driven-recovery diagnostic amplitude and
far simpler than the individual signed rows
`[-109.9, -43.4]`, `[125.0, 49.2]`, and `[-6.90, -3.73]`.  The emerging
constraint is therefore: the static source-moment amplitude may be compact,
while the unresolved physics is how that amplitude is distributed across
geometry-controlled recovery paths without fitting to empymod.

That summed-amplitude clue is now auditable without another least-squares fit.
The trace-kernel CLI accepts either physical prescribed coefficients or
dimensionless values through

```text
--prescribed-normalized-coefficients <values>
```

where normalized values are multiplied by `mu * delta_sigma * L^2` before
evaluation.  Three prescribed checks were generated from the same saved
`delta_residual` trace:

```text
outputs/source_history_hj_rise_decay_kernel_prescribed_geometry_sum_single_tau.json
  fast_tau = 5.6330878831955205e-5 s
  normalized coefficients = [8.1876, 2.0719]
  prescribed trace relative L2 = 0.1731

outputs/source_history_hj_rise_decay_kernel_prescribed_geometry_sum_single_tau_rounded.json
  fast_tau = 5.6330878831955205e-5 s
  normalized coefficients = [8, 2]
  prescribed trace relative L2 = 0.1808

outputs/source_history_hj_rise_decay_kernel_prescribed_geometry_sum_empirical_tau.json
  fast_tau = 0.11 ms
  normalized coefficients = [8.1876, 2.0719]
  prescribed trace relative L2 = 0.1791
```

The first case is close to the best single geometry-tau fit (`0.1637`), but it
does not approach the three-path fitted geometry-tau result (`0.0389`).  The
empirical `0.11 ms` tau only reaches `0.1218` when its coefficients are fitted
separately (`[9.319, 2.677]` normalized).  Therefore the coefficient-sum
collapse is a real static-amplitude clue, not a complete source-history law:
the remaining derivation must compute how the compact amplitude is distributed
across recovery paths and effective time constants.

A non-fitted moment-constraint distribution was then tested in the runtime
driven-relaxation basis.  It uses the three geometry-derived response times

```text
4.523893420572e-5 s, 5.63308788319552e-5 s, 2.0032567403565607e-4 s
```

and assigns weights without reading the residual trace.  Two variants were
audited:

```text
sum(w) = 1, sum(w/tau_R) = 0
sum(w) = 1, sum(w/tau_R) = 0, sum(w/tau_R^2) = 0
```

The first is the minimum-norm solution satisfying amplitude preservation and
initial-slope cancellation.  The second also cancels the next continuous-time
early moment.  The summary report is

```text
outputs/source_history_hj_driven_relaxation_geometry_tau_moment_constraint_summary.json
```

Using the rounded compact amplitude `[8, 2]`, the slope-cancel candidate gives
trace L2 `0.18877`; using the unrounded compact amplitude
`[8.1876, 2.0719]` gives `0.18639`.  The slope+curvature candidate is worse
(`0.28479` rounded, `0.28273` unrounded).  Both are far from the fitted
geometry-tau three-path result (`0.03894`).  Therefore the fitted row weights
are not explained by simple interpolation, compact-amplitude preservation, or
low-order early-time moment cancellation over the geometry-derived response
times.  The missing source-history law still needs the actual FV/HJ source and
MMR recovery projection that produces the signed modal distribution.

A second prescribed check tested whether the signs come from simple nested
support shells or inclusion-exclusion over the three geometry tau groups.  The
summary is

```text
outputs/source_history_hj_driven_relaxation_geometry_tau_shell_weight_summary.json
```

The baseline driven-relaxation single geometry tau with rounded amplitude
`[8, 2]` gives trace L2 `0.20023`.  The simplest shell difference
`[-1, 2, 0] * [8, 2]`, interpreted as `-small + 2*intermediate`, improves to
`0.18106`.  The three-level alternatives `[-1, 3, -1]` and `[1, -2, 2]` are
much worse (`0.40498` and `0.33182`).  Using the unrounded compact amplitude
with `[-1, 2, 0]` gives `0.19341`.

This weak positive result is useful but not sufficient.  A low-order shell
difference carries some recovery-path time-shape information, but it remains
far above the fitted three-geometry-tau result (`0.03894`) and does not produce
the required source-moment-2 trace.  The next derivation should therefore look
for the actual FV/HJ source-to-recovery projection that weights shell-like
paths, rather than using hand-picked inclusion-exclusion coefficients.

The same shell-weight audit was then moved from idealized tau columns to the
actual projected driven-response reports.  The trace-audit CLI now accepts

```text
--combine-report-weights <w0> <w1> ...
```

which linearly combines the selected `source_moment_projection.coefficients`
from multiple driven-response reports before applying the same fitted-amplitude
and compact-amplitude trace audit.  Using

```text
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_sr0_rr0_pad0.json
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_sr1_rr1_pad0.json
outputs/local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_sr1_rr1_pad1.json
```

against the sub5 clean `delta_residual` target gives the summary

```text
outputs/local_recovery_spectrum_relaxdiff_actual_driven_shell_trace_audit_summary.json
```

The actual `[-1, 2, 0]` projected shell is stronger than the idealized tau
shell: fitted trace L2 `0.16364`, compact `[8,2]` trace L2 `0.16497`, and
fitted normalized amplitude `[7.995, 1.829]`.  The three-level alternatives are
rejected: `[-1, 3, -1]` gives fitted/compact L2 `0.28280/0.40580`, and
`[1, -2, 2]` gives `0.41793/0.51295`.  However even the best actual shell
does not generate the required early-time behavior.  At the first selected
gate, the compact shell predicts `[5.19, 1.39]` while the target trace is
`[1.34, -0.22]` in `mu*delta_sigma*L^2` units.  Therefore the missing law is
not simply nested-support inclusion-exclusion.  It must derive a dynamic
source/MMR recovery projection that both suppresses the earliest gates and
produces the signed moment-2 response.

Holding the compact source-moment amplitude fixed gives a sharper diagnostic
target for the remaining weighting law.  With

```text
--fit-compact-report-weights --compact-normalized-amplitude 8 2
```

the same three actual reports produce

```text
outputs/local_recovery_spectrum_relaxdiff_actual_driven_compact_report_weight_fit_trace_audit.json
```

and reach trace L2 `0.08332` with fitted report weights
`[-6.666, 7.627, 0.131]` and condition number about `197`.  This is much better
than the simple `[-1, 2, 0]` shell, but it is not a production closure: the
weights come from residual fitting, the first two supports cancel strongly, and
the first selected gate remains `[2.53, 0.99]` versus target `[1.34, -0.22]`.
The useful constraint is therefore that the non-fitted FV/HJ source-recovery
operator must generate a signed, cancellation-heavy support weighting while
also adding the missing early-time moment-2 sign change.

The fitted report weights become simpler in independent shell coordinates.  For
nested reports `R0 ⊂ R1 ⊂ R2`, define shells

```text
S0 = R0,  S1 = R1 - R0,  S2 = R2 - R1.
```

The fitted report weights `[-6.666, 7.627, 0.131]` correspond to shell weights
`[1.092, 7.758, 0.131]`.  Rounding this clue gives the integer shell candidate
`[1, 8, 0]`, or report weights `[-7, 8, 0]`.  Its trace audit is saved in

```text
outputs/local_recovery_spectrum_relaxdiff_actual_driven_shell_mid8_trace_audit.json
```

and gives compact `[8,2]` trace L2 `0.10631`, compact after one optimal scalar
`0.08426`, and fitted two-amplitude trace L2 `0.05291` with normalized
amplitude `[8.726, 1.608]`.  This nearly matches the fitted compact
report-weight diagnostic without carrying arbitrary real-valued report weights,
so `[1,8,0]` is the sharpest simple support-weight clue so far.

It still does not transfer as a production receiver correction.  The
corresponding sub5 `time_series_source_moments` runtime replay uses

```text
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_relaxdiff_actual_shell_mid8_time_series_compact.yaml
examples/hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_relaxdiff_actual_shell_mid8_time_series_fitted_amp.yaml
outputs/hj_relaxdiff_actual_shell_mid8_runtime_summary.json
```

After receiver-only postprocessing of the sub5 H/J result, the compact
amplitude gives `Hz` relative L2 `0.228/0.250/0.228`, and the trace-fitted
amplitude gives `0.229/0.252/0.230`.  This is a real improvement over the raw
H/J `current_biot` `Hz` errors, but it is worse than the trace-fitted
multi-tau runtime check (`0.171/0.176/0.172`) and far weaker than the trace
L2 alone suggests.  Therefore the next missing piece is not only a support-shell
time trace.  It is the finite-volume source/MMR recovery-to-receiver operator
that maps that trace into the three near-source `Hz` columns.

The sub5 runtime error was then decomposed using the matched IP/no-IP sampled
reports and the corrected validation reports:

```text
outputs/hj_relaxdiff_actual_shell_mid8_error_decomposition.json
```

The sign convention in the decomposition is `error = numerical - reference`.
For an IP-only source-history correction that vanishes when `delta_sigma = 0`,
the clean target added to the IP numerical data is

```text
clean_delta_correction = noip_error - raw_ip_error.
```

The final corrected IP error is therefore

```text
corrected_ip_error = noip_error
                   + (actual_correction - clean_delta_correction).
```

This identity is now generated by `atem3d.clean_delta_decomposition_cli` so
candidate runtime reports can be compared under the same clean-delta metric.
For the `[-7,8,0]` compact sub5 replay,
`outputs\hj_debye_ip_sub5_relaxdiff_actual_shell_mid8_compact_clean_delta_decomposition.json`
records a small correction mismatch:

```text
Hz x=-20/0/20 mismatch relative L2 = 0.026/0.037/0.026
aggregate actual_correction_relative_l2_to_ideal = 0.0759
```

The trace-fitted-amplitude variant
`outputs\hj_debye_ip_sub5_relaxdiff_actual_shell_mid8_fitted_amp_clean_delta_decomposition.json`
is cleaner in IP-only space:

```text
Hz x=-20/0/20 mismatch relative L2 = 0.0169/0.0157/0.0171
aggregate actual_correction_relative_l2_to_ideal = 0.0421
```

but its final receiver-space `Hz` L2 is slightly worse than the compact replay,
so receiver-space performance is now limited by more than clean-delta trace
accuracy.  For comparison, the older one-mode driven-recovery replay
`outputs\hj_debye_ip_sub5_driven_recovery_clean_delta_decomposition.json` has
larger mismatch `0.063/0.072/0.063`, and the no-IP source-cell law has
`actual_correction_relative_l2_to_ideal` about `1.24/1.27/1.24`, confirming that
it is not an IP clean-delta term.

The no-IP/current-Biot baseline measured against the IP reference norm is
already

```text
Hz x=-20/0/20 no-IP relative L2 = 0.214/0.239/0.214.
```

Thus the final `0.228/0.250/0.228` runtime errors are mostly the no-IP magnetic
recovery baseline plus a small constructive clean-delta mismatch.  `[1,8,0]`
and the trace-fitted amplitude are real IP-delta improvements.  However a
source-history term that correctly vanishes for `delta_sigma = 0` cannot remove
the no-IP baseline.  The full production solver now has two coupled but
distinct remaining requirements: derive the non-fitted IP clean-delta
source-history law, and reduce the underlying no-IP H/J current-Biot near-source
magnetic recovery error.

Those fitted geometry-tau coefficients were also pushed through the runtime
path in
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_geometry_tau_multimode.yaml`.
The config uses one `driven_recovery_source_moments` term with

```text
response_taus = [4.5239e-5, 5.6331e-5, 2.0033e-4] s
source_moment_degrees = [0, 2]
```

and the trace-fitted coefficient table from
`outputs\source_history_hj_rise_decay_kernel_from_recovery_sweep_tau_estimates.json`.
Receiver-only postprocessing of
`outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_data_only.h5`
produced
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_geometry_tau_multimode_data_only.h5`.
The empymod comparison
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_geometry_tau_multimode_t010_1ms_validation.json`
gives `Hz` relative L2 about `0.171/0.176/0.172`, improving strongly over the
uncorrected `0.588/0.650/0.589` and slightly over the previous single driven
diagnostic.  This validates the multi-`response_taus` runtime path, but it does
not close the derivation because the coefficient table is still fitted from the
trace residual.

## 12. No-IP H/J Source-Diffusion Baseline

This section separates the no-IP H/J `current_biot` baseline from the IP-only
clean-delta source-history problem.  It is important because an IP production
term that vanishes when `delta_sigma = 0` cannot remove the no-IP near-source
`Hz` error.

For the no-IP H/J step, the code implements the discrete equations

```text
A h^{n+1} = rhs
A = C^T M_rho C + M_mu/dt_n
rhs = M_mu h^n/dt_n + C^T M_rho s_f^{n+1}
```

where `C = mesh.edge_curl`, `M_rho` is the face resistivity inner product, and
`M_mu` is the edge permeability inner product.  With Debye terms present, the
same structure becomes

```text
A = C^T M_rho_eff C + M_mu/dt_n
rhs = M_mu h^n/dt_n
    + C^T M_rho_eff s_f^{n+1}
    - C^T M_unit E_hist^n.
```

The H/J source sign convention in the runtime is

```text
s_f(t) = -source.face_vector_at(mesh, t)
s_0    = -source.initial_face_vector(mesh)
J_c^{n+1} = C h^{n+1} - s_f^{n+1}.
```

For the ideal long-on step-off source used in the current validation configs,
`s_f^{n+1} = 0` for every positive time node.  Therefore the positive-time
no-IP H/J solve has no explicit source term in the RHS; the source enters
through the long-on DC/MMR initial state `h^0`.  The receiver-side
`current_biot` reconstruction then evaluates

```text
H_R(t, r) = B_R [C h(t) - s_f(t)].
```

At `t > 0`, this is simply `B_R C h(t)`.  The missing no-IP baseline is
therefore not a missing positive-time impressed source in the H/J time step.
The source-neighborhood audits show that it behaves like a receiver-side
source-neighborhood transfer error whose leading static shape is the raw
degree-zero H/J face source moment,

```text
v_0 = s_0.
```

The local coupling routine `source_face_moment_basis` does not hide a
normalization in this vector: degree zero is the signed source vector itself
on the active faces.  This fixes the diagnostic source-diffusion ansatz as

```text
Delta H_R(t, r) = a(t) B_R v_0(r),
a(t) = A exp[-(t - t0)/(m tau0)],
tau0 = mu * sigma_midpoint * L^2,
A = a0 * tau0.
```

The current cross-grid evidence is a constraint, not a derivation:

```text
a0 = A/(mu sigma L^2) ~= -3.6
m  ~= 1.25 to 1.5
t0 = first fitted sample time, currently 1.0e-4 s
```

The global replay config uses `a0 = -3.467727352856916` and `m = 1.5`.  It
reduces the no-IP H/J `Hz` errors to the few-percent range on the two checked
source-centered grids, but both constants remain fitted from source-neighborhood
reports.  They cannot be promoted to production physics until `a0` and `m`
are obtained from a finite-volume H/J source/MMR transfer model or an
independently verified local/global source-neighborhood operator.

The new geometry audit entry point records the source-vector metrics needed to
continue that derivation:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_geometry_audit_cli `
  examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_global_law.yaml `
  -o outputs\hj_noip_source_centered_current_biot_sub5_source_geometry_audit.json
```

It reports the runtime source-vector sign/location, active face counts,
orientation counts, face-area weighted norms, midpoint cell widths, midpoint
conductivity, `tau0 = mu sigma L^2`, and any configured
`source_diffusion_kernel_source_moments` amplitude.  For H/J `current_biot`
configs it also reports the static receiver response `B_R v_0`.  These
quantities are the next audit surface for explaining why the fitted normalized
amplitude is close to `-3.5` and how the time multiplier depends on the
source-cell geometry.

The first two geometry audits are now saved as

```text
outputs\hj_noip_source_centered_current_biot_sub5_source_geometry_audit.json
outputs\hj_noip_source_centered_zrefined_current_biot_sub5_source_geometry_audit.json
```

Both configs have `L = 50 m`, `sigma_midpoint = 0.01 S/m`,
`tau0 = 3.141592653175e-5 s`, and 11 active x-oriented H/J source faces.  The
source midpoint cell widths are `[5, 5, 1] m` on the original grid and
`[5, 5, 0.5] m` on the z-refined grid.  As expected for an area-normalized H/J
face source, the raw signed source-vector scale changes with the face area:

```text
original:  L1 = 2.2,  L2 = 0.6633, area-weighted L1 = 11.0
z-refined: L1 = 4.4,  L2 = 1.3266, area-weighted L1 = 11.0
```

The H/J face inner-product norms are not themselves invariant:

```text
original:  ||v_0||_Munit = 3.3166, ||v_0||_Mrho = 33.1662,
           ||C^T M_rho v_0||_2 = 2.3917e3
z-refined: ||v_0||_Munit = 4.6904, ||v_0||_Mrho = 46.9042,
           ||C^T M_rho v_0||_2 = 4.7138e3
```

The quantity that stays nearly invariant is the actual receiver-side static
shape used by the diagnostic correction:

```text
original  B_R v_0 Hz(x=-20,0,20) =
  [0.0127153, 0.0152581, 0.0127153], ||B_R v_0||_2 = 0.0235832
z-refined B_R v_0 Hz(x=-20,0,20) =
  [0.0127248, 0.0152689, 0.0127248], ||B_R v_0||_2 = 0.0236005
```

Thus the cross-grid stability of `A/(mu sigma L^2)` is not explained by raw
face-vector norms or by the local `C^T M_rho v_0` RHS norm.  It is consistent
with an area-integrated source face vector after projection through the same
receiver matrix `B_R` used by the runtime correction.  The next derivation
should therefore target the composite source-neighborhood transfer
`B_R T_FV/MMR v_0`, not a standalone source-vector norm.

The same geometry audit can now run over named config overrides.  Applied to
the source-cell-thickness sweep,

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_geometry_audit_cli `
  examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_global_law.yaml `
  --sweep-cases examples\validation_sweep_hj_noip_source_cell_thickness.yaml `
  -o outputs\validation_sweep_hj_noip_source_cell_thickness_source_geometry_audit.json
```

the static source and receiver metrics are

```text
source cell hz   raw L1   area L1   ||C^T M_rho v_0||_2   ||B_R v_0||_2
1.00 m            2.2      11.0      2.3917e3              2.3583e-2
0.75 m            2.9333   11.0      3.1619e3              2.3593e-2
0.50 m            4.4      11.0      4.7138e3              2.3600e-2
0.25 m            8.8      11.0      9.3926e3              2.3605e-2
```

This rules out another tempting shortcut: the source-cell-thickness dependence
in the no-IP H/J `Hz` error is not caused by the static receiver matrix
`B_R v_0`, which is essentially invariant over this sweep.  The remaining
`tau_multiplier` question is therefore a dynamic diffusion/operator issue:
derive how the finite-volume source-neighborhood transfer evolves in time
before it is projected by `B_R`, rather than rescaling the static source shape.

Two missing full-field source-cell-thickness cases were then generated:

```text
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p75_current_biot_sub5.yaml
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p25_current_biot_sub5.yaml
outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p75_current_biot_sub5_full.h5
outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p25_current_biot_sub5_full.h5
```

Their empymod sample reports are

```text
outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p75_current_biot_sub5_t010_1ms_samples_report.json
outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p25_current_biot_sub5_t010_1ms_samples_report.json
```

and the matching source-neighborhood audits are

```text
outputs\hj_noip_source_centered_sourcecell0p75_current_biot_sub5_source_neighborhood_audit.json
outputs\hj_noip_source_centered_sourcecell0p25_current_biot_sub5_source_neighborhood_audit.json
```

The 0.75 m case has base/source-neighborhood-fit combined `Hz` L2
`0.1809 -> 0.0120`; the 0.25 m case has `0.1801 -> 0.00590`.  Therefore the
same source-neighborhood subspace explains the no-IP residual across the
source-cell-thickness sweep.

The four-report law audit

```text
outputs\hj_noip_source_centered_source_cell_thickness_source_diffusion_law_audit.json
```

uses the 1.00 m, 0.75 m, 0.50 m, and 0.25 m source-cell cases.  The per-report
best active-source exponential fits are

```text
source cell hz   best m   A/(mu sigma L^2)   coefficient L2
1.00 m           1.25     -3.522             0.113
0.75 m           1.25     -3.613             0.122
0.50 m           1.50     -3.671             0.127
0.25 m           1.25     -3.610             0.077
```

The normalized amplitude mean is `-3.604` with coefficient of variation about
`1.48%`; the first-gate normalized coefficient mean is `-3.570` with CV about
`2.40%`.  This is stronger evidence that the amplitude normalization
`A = a0 mu sigma L^2` is real.  The time multiplier is less settled: the best
single global law over all four reports is `m = 1.25`,
`A/(mu sigma L^2) = -3.667`, combined coefficient L2 `0.132`, while `m = 1.5`
is close at `0.138`.  The production derivation should therefore prioritize
deriving `a0` first, and treat `m` as a dynamic finite-volume diffusion
constant that remains weakly constrained between roughly `1.25` and `1.5`.

That four-report law was also replayed through the receiver postprocess path
with

```text
normalized_amplitude = -3.6668447223301657
tau_multiplier = 1.25
amplitude_time = 1.0e-4 s
```

using the diagnostic configs

```text
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_source_cell_law.yaml
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p75_current_biot_sub5_diagnostic_source_diffusion_source_cell_law.yaml
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_zrefined_current_biot_sub5_diagnostic_source_diffusion_source_cell_law.yaml
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p25_current_biot_sub5_diagnostic_source_diffusion_source_cell_law.yaml
```

The postprocessed empymod comparisons give `Hz` relative L2:

```text
source cell hz   Hz x=-20      Hz x=0       Hz x=20
1.00 m           0.0200        0.0304       0.0205
0.75 m           0.0229        0.0286       0.0237
0.50 m           0.0404        0.0327       0.0381
0.25 m           0.0151        0.0232       0.0161
```

Thus the four-report global law transfers to receiver data across the tested
source-cell-thickness sweep, reducing the no-IP H/J `current_biot` baseline
from about `0.18-0.20` to a few percent.  It remains diagnostic-only: the
0.50 m z-refined case still prefers `m = 1.5` in coefficient space, and the
constants are constrained by empymod/source-neighborhood residuals rather than
derived from the finite-volume H/J transfer operator.

The driven-response trace audit can now use this no-IP source-neighborhood
target directly:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.recovery_driven_trace_audit_cli `
  outputs\local_recovery_noip_source_centered_degree0_debye_decay_m1_driven_response.json `
  --target-source-neighborhood-report outputs\hj_noip_source_centered_current_biot_sub5_source_neighborhood_audit.json `
  --compact-normalized-amplitude -3.6668447223301657 `
  -o outputs\local_recovery_noip_source_centered_degree0_debye_decay_m1_driven_active_source_audit.json
```

For `--target-source-neighborhood-report`, the audit reads
`candidate_static_fits[active_source].per_time_coefficients` and normalizes
amplitudes by `source_diffusion_time_s = mu sigma_midpoint L^2`, rather than by
`mu delta_sigma L^2`.  This lets the same non-fitted driven-response machinery
test the no-IP baseline without pretending it is an IP clean-delta term.

Using the 210-cell source/receiver support, one degree-zero face source moment,
and a simple `debye_decay` driver with `driver_tau = m tau0`, the single-report
active-source trace gives:

```text
m       fitted trace L2   fitted amplitude/tau0   compact -3.6668 L2
0.50    0.3387            -574.38                 0.9944
0.75    0.2009            -134.24                 0.9738
1.00    0.1210            -57.41                  0.9371
1.25    0.1272            -32.41                  0.8888
1.50    0.1820            -21.32                  0.8343
2.00    0.2976            -11.84                  0.7231
3.00    0.4661            -5.81                   0.5687
4.00    0.5754            -3.71                   0.5755
```

This is a useful constraint but not a closure.  The driven-response shape with
`m = 1.0` is as good as the empirical exponential law for the 1.00 m case
(`0.121` coefficient L2), but the amplitude scale is not derived: the compact
`-3.6668 tau0` amplitude is too small by a factor of about `15.7`.

The same `m = 1.0` driven-response shape was then checked over the
source-cell-thickness sweep:

```text
source cell hz   fitted trace L2   fitted amplitude/tau0
1.00 m           0.1210            -57.41
0.75 m           0.1341            -58.98
0.50 m           0.1894            -63.84
0.25 m           0.0968            -58.93
```

The fitted amplitude is now much more stable than in the single-report `m`
scan, which means the driven local operator is capturing part of the transferable
time shape.  The remaining gap is also clear: a production law still has to
derive the large normalization factor that maps the local driven coefficient
back to the physical source-neighborhood correction, and it still has to explain
why the z-refined 0.50 m case is weaker in this operator audit.

That large normalization factor is now audited against both source geometry and
the first selected driven-response gate with

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_diffusion_amplitude_scale_audit_cli `
  outputs\local_recovery_noip_source_centered_cell1p00_degree0_debye_decay_m1_driven_active_source_audit.json `
  outputs\local_recovery_noip_source_centered_cell0p75_degree0_debye_decay_m1_driven_active_source_audit.json `
  outputs\local_recovery_noip_source_centered_cell0p50_degree0_debye_decay_m1_driven_active_source_audit.json `
  outputs\local_recovery_noip_source_centered_cell0p25_degree0_debye_decay_m1_driven_active_source_audit.json `
  --geometry-report outputs\validation_sweep_hj_noip_source_cell_thickness_source_geometry_audit.json `
  --case-keys source_cell_1p00m source_cell_0p75m source_cell_0p50m source_cell_0p25m `
  -o outputs\local_recovery_noip_source_centered_source_cell_m1_amplitude_scale_audit.json
```

For this source geometry, `L = 50 m`, the along-wire core cell width is `dx = 5
m`, and the active face count is `11`, i.e. `10` source intervals.  The simple
geometry scale

```text
S_geom = (pi/2) * L/dx = (pi/2) * (active_count - 1) = 15.707963...
```

matches the fitted compact-scalar bridge surprisingly well, but the augmented
audit shows a more direct explanation: the compact source-diffusion law is
normalized to the first target gate (`0.1 ms`), while the driven response uses a
Debye-like driver that starts decaying at `t = 0`.  For this run,
`tau0 = 3.141592653175e-5 s = pi * 1e-5 s`, `dt = 1e-5 s`, and the first target
gate is step 10, so the BE history value is

```text
g0(0.1 ms) = (tau0 / (tau0 + dt))^10 = 0.0630703,
1/g0 = 15.8553.
```

The real driven projection has first selected responses `0.06308-0.06322`,
whose inverses are `15.82-15.85`.  The trace audit now also reports
`compact_first_gate_scalar`, computed from the first selected target gate only,
and `compact_first_gate_relative_l2`, the full-window error after applying that
first-gate scalar.  The report therefore separates the full-window optimal
scalar from the pure first-gate time-basis bridge:

```text
source cell hz   opt scalar   opt/(1/first response)   first-gate scalar   first-gate L2   first-gate/(1/first response)
1.00 m           15.6560      0.9898                   15.0342             0.1273          0.9504
0.75 m           16.0839      1.0146                   15.3206             0.1421          0.9665
0.50 m           17.4095      1.0982                   16.0525             0.2043          1.0126
0.25 m           16.0719      1.0138                   15.2981             0.1080          0.9650
```

Thus the earlier `(pi/2)*(L/dx)` match is likely a numerical coincidence for
this mesh/time grid: `(1 + dt/tau0)^10` is close to `5*pi` because
`tau0 ~= pi*dt` and the first target gate is 10 steps after shutoff.  The
stronger conclusion is that the no-IP driven-response amplitude mismatch is
mostly a time-origin/first-gate normalization issue, not yet evidence for a
line-source segment-count factor.  First-gate scaling reduces the compact trace
L2 from about `0.94` to `0.108-0.204`, close to the fitted-amplitude range
`0.097-0.189`, but it does not remove the remaining shape error.  The
first-gate scalar is not exactly `1/first_response` because the compact
amplitude `-3.6668447223` is not exactly the first target coefficient in every
source-cell case; the mean first-gate scalar ratio is `0.974` against
`1/first_response` and `0.982` against `S_geom`.  A production derivation must
align the source-diffusion ansatz and the driven operator on the same BE time
basis before interpreting any remaining geometry factor, and must still derive
the residual source-neighborhood shape evolution rather than fitting it.

The trace audit now also records per-time relative errors and per-time error
energy fractions for the raw compact trace, the all-window optimal scalar, and
the first-gate scalar.  This rules out the simple explanation that the 0.50 m
case is only a late-time small-target artifact.  With first-gate scaling, the
largest single-gate error-energy fractions are small and occur in the early
middle window:

```text
source cell hz   first-gate L2   peak error gate   peak fraction
1.00 m           0.1273          0.140 ms          5.2%
0.75 m           0.1421          0.140 ms          5.6%
0.50 m           0.2043          0.150 ms          6.3%
0.25 m           0.1080          0.140 ms          9.9%
```

Thus the remaining no-IP source-neighborhood mismatch is a distributed
early-to-middle time-shape error after time-origin alignment.  The 0.50 m
z-refined case remains the outlier, so the next non-fitted derivation should
test the driven local operator's recovery-time distribution and support
restriction, not another static source-vector or geometry scalar.

That test was run by regenerating only the 0.50 m z-refined local
`debye_decay` driven response with `driver_tau = m tau0`, keeping the same
210-cell source/receiver support and the same source-neighborhood target:

```text
m      fitted/optimal L2   first-gate L2   optimal scalar   first-gate scalar   peak error gate
0.50   0.4094              0.4422          170.874          139.579             0.130 ms
0.75   0.2819              0.3108          40.317           34.811              0.140 ms
1.00   0.1894              0.2043          17.409           16.053              0.150 ms
1.25   0.1357              0.1364          9.925            9.786               0.400 ms
1.50   0.1285              0.1386          6.591            6.937               0.130 ms
2.00   0.1927              0.2673          3.730            4.434               0.150 ms
3.00   0.3320              0.5489          1.897            2.776               0.160 ms
```

The 0.50 m outlier is therefore not explained by static source-cell geometry
alone: a slower driven recovery time near `1.25-1.5 tau0` removes most of the
extra shape error.  The large variation of the required scalar, however, means
this is still only a diagnostic recovery-time constraint.  A production law must
derive how the FV/H/J source-neighborhood operator distributes recovery times
with mesh/support restriction, while keeping amplitude and time-origin
normalization on the same BE basis.

The same `m` sweep was then completed for the 1.00 m, 0.75 m, 0.50 m, and
0.25 m source-cell targets.  The per-case optima are:

```text
source cell hz   best m   fitted/optimal L2   first-gate L2   optimal scalar   first-gate scalar
1.00 m           1.00     0.1210              0.1273          15.656           15.034
0.75 m           1.25     0.1278              0.1305          9.095            9.340
0.50 m           1.50     0.1285              0.1386          6.591            6.937
0.25 m           1.25     0.0834              0.0872          9.092            9.326
```

A single global `m = 1.25` is the best compromise over this sweep:

```text
m      mean fitted L2   max fitted L2   mean first-gate L2   optimal scalar mean/std
0.50   0.3608           0.4094          0.3950               162.051 / 5.314
0.75   0.2244           0.2819          0.2528               37.997 / 1.392
1.00   0.1353           0.1894          0.1454               16.305 / 0.660
1.25   0.1185           0.1357          0.1216               9.238 / 0.410
1.50   0.1567           0.1820          0.1830               6.097 / 0.294
2.00   0.2597           0.2976          0.3558               3.408 / 0.191
3.00   0.4214           0.4661          0.6727               1.693 / 0.121
```

This preserves the earlier empirical source-diffusion observation that a global
time multiplier near `1.25` transfers well, but now in the local driven-operator
trace space.  It also sharpens the missing derivation: for a fixed `m` the
optimal scalar is fairly stable across source-cell thickness, but the scalar
changes by orders of magnitude as `m` changes.  Therefore amplitude and recovery
time cannot be chosen independently in the production H/J law; both must come
from the same discrete FV/MMR source-neighborhood operator.

The local positive magnetic-diffusion spectra were then regenerated for the
same source/receiver support (`source_cell_radius = receiver_cell_radius = 1`,
`padding = 0`, `max_modes = 24`) to check whether `m tau0` is simply the slowest
positive eigen-time of the current local Dirichlet support:

```text
source cell hz   slowest positive tau   tau0 / tau_slow   best_m*tau0 / tau_slow
1.00 m           2.084e-08 s            1507.7            1507.7
0.75 m           7.270e-07 s            43.2              54.0
0.50 m           7.270e-07 s            43.2              64.8
0.25 m           7.270e-07 s            43.2              54.0
```

Each spectrum discards `359` near-null modes and keeps the smallest positive
eigenvalues.  Thus the empirical `m tau0` scale is much slower than the bare
positive spectrum of this small local support.  The current driven-response
shape constraint is therefore not a direct "pick the slowest local eigenmode"
rule; it is a source-neighborhood transfer timescale produced by the way the
grounded-source forcing, local support boundary, and receiver/source-moment
projection interact on the BE grid.  This is an important negative result for
the production derivation.

Padding the 0.50 m z-refined support then tested whether that timescale is just
a small-box boundary artifact.  The same `m = 1.25` and `m = 1.5`
driven-response audits were repeated with `padding = 0..3`, increasing the local
support from `210` cells / `906` edge dofs to `1980` cells / `6978` edge dofs:

```text
m      padding   fitted L2   first-gate L2   optimal scalar
1.25   0         0.135681    0.136391        9.924968
1.25   1         0.135681    0.136391        9.906255
1.25   2         0.135681    0.136391        9.876096
1.25   3         0.135681    0.136391        9.837377
1.50   0         0.128507    0.138648        6.590638
1.50   1         0.128507    0.138648        6.578821
1.50   2         0.128507    0.138648        6.559634
1.50   3         0.128507    0.138648        6.534989
```

The normalized time shape is essentially invariant under this padding sweep,
and the amplitude scalar drifts by only about one percent.  Therefore the
remaining no-IP driven-transfer timescale is also not explained by simply
expanding the rectangular local support.  The next derivation should inspect the
source forcing/projection and BE driver coupling that define the
source-neighborhood transfer operator.

The first forcing/projection check used the 0.50 m z-refined case at the global
compromise `m = 1.25`, comparing the raw baseline against the existing
charge-conserving source projection, charge-conserving MMR initial states, and
global-MMR steady forcing:

```text
variant                         fitted L2   first-gate L2   optimal scalar   first-gate scalar
raw baseline                    0.135681    0.136391        9.924968         9.785724
charge_conserving               0.135681    0.136391        9.924968         9.785724
charge_conserving_mmr_initial   0.135681    0.136391        9.924968         9.785724
global_cc_mmr_initial           0.135681    0.136391        9.924968         9.785724
global_mmr_steady_forcing       0.135681    0.136391        9.918680         9.779524
```

The full coefficient matrices for the MMR-initial variants differ from the raw
baseline at the `t = 0` sample, but over the selected target window
`0.1-1.0 ms` they match the raw driven coefficients to roundoff.  The
`global_mmr_steady_forcing` variant changes the selected coefficient trace by
only relative L2 `6.34e-4`.  Thus the current source-projection and initial-MMR
switches do not generate the missing source-neighborhood transfer timescale.
The next production-facing derivation must work at the discrete source
forcing/history operator itself, not at these already-tested projection toggles.

The driven-trace audit now records a `driver_follow_relative_l2` diagnostic:
each selected source-moment coefficient trace is normalized by its first selected
value and compared with the selected `driver_values` normalized the same way.
For the 0.50 m z-refined `m` sweep, the result is essentially exact driver
following:

```text
m      fitted L2   driver-follow relative L2
0.50   0.4094      3.19e-16
0.75   0.2819      1.14e-15
1.00   0.1894      5.48e-16
1.25   0.1357      1.08e-15
1.50   0.1285      8.76e-16
2.00   0.1927      6.01e-16
3.00   0.3320      1.21e-15
```

This is the decisive interpretation of the local driven-response sweep: over
the `0.1-1.0 ms` target window, the local H/J solve is quasi-static relative to
the imposed Debye driver, so it supplies a spatial/static transfer amplitude
while the time shape is inherited from the externally prescribed driver history.
Therefore the empirical `m` is not generated by the local magnetic-diffusion
solve used in this diagnostic.  The missing production term is now localized to
the discrete grounded-source history kernel that should drive the
source-neighborhood transfer, including its first-gate normalization and its
coupling to the receiver-projected source moment.

The source-diffusion law audit was updated accordingly with
`--basis-kind be_decay`, which uses the first-gate-normalized backward-Euler
Debye decay on the sampled time grid instead of the previous continuous
`exp(-(t-t0)/tau)` basis.  On the four source-cell source-neighborhood reports:

```text
basis        best global m   A/tau0      combined coefficient L2
continuous   1.25            -3.666845   0.131752
be_decay     1.25            -3.506950   0.128617
```

The BE per-report best fits now line up with the driven-response `m` sweep:

```text
source cell hz   best m   A/tau0      coefficient L2
1.00 m           1.00     -3.629264   0.120993
0.75 m           1.25     -3.451166   0.127756
0.50 m           1.50     -3.527946   0.128507
0.25 m           1.25     -3.449649   0.083377
```

This removes the earlier continuous-vs-BE ambiguity: when the source-neighborhood
coefficient traces are audited in the same BE time basis as the driven reports,
the local driven solve adds no independent time shape.  The remaining unknown is
a non-fitted derivation of the BE grounded-source history kernel and its
amplitude normalization, not another local recovery solve.

The runtime/postprocess `source_diffusion_kernel_source_moments` hook now mirrors
that audit basis through `basis_kind: be_decay`.  The default
`basis_kind: continuous` preserves the older exponential replay; `be_decay`
computes the order-zero BE Debye history from the configured `time_steps` and
normalizes it at `amplitude_time`.  This makes the diagnostic replay consistent
with the source-neighborhood BE audit, but it still does not derive the
production H/J source-history/MMR law or the IP clean-delta coupling.

The corresponding no-IP runtime validation is saved as:

```text
examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_source_cell_law_be_decay.yaml
outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_source_cell_law_be_decay_t010_1ms_validation.json
```

It records `basis_kind: be_decay` in the validation metadata and gives `Hz`
relative L2 `0.0224/0.0323/0.0230` over `0.1-1 ms`.  This is a runtime replay
check for the BE diagnostic basis, not a production closure.

`empymod_validation` source-history metadata now also records `requires_ip`.
This is `false` for the no-IP `source_diffusion_kernel_source_moments` baseline
diagnostic and `true` for prescribed, driven, initial-polarization, and future
IP clean-delta candidates.  The field is an audit guard: a production IP
source-history/MMR correction must be marked IP-dependent and must vanish when
`delta_sigma = 0`.  The EB and H/J runtime paths now enforce the same rule by
filtering `requires_ip` corrections whenever the active IP model has no positive
Debye `delta_sigma`, not merely when the Debye term list is empty.
For prescribed and driven source-moment diagnostics, `normalized_coefficients`
are therefore only a dimensional replay convenience: the runtime multiplies them
by `mu * sum(delta_sigma_source) * source_length^2`, but the coefficients still
have to come from a derivation or an explicitly diagnostic report.

The same no-IP baseline law was then applied alone to the Debye-IP sub5 H/J
`current_biot` result, using

```text
examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_noip_source_cell_law.yaml
outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_noip_source_cell_law_data_only.h5
outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_noip_source_cell_law_t010_1ms_validation.json
```

with empymod run in `--use-config-ip` mode.  The raw IP sub5 report has
`Hz` L2 `0.4135/0.4755/0.4144`; adding only the no-IP source-cell law gives
`0.3948/0.4664/0.3955`.  This small improvement is important evidence: the
no-IP baseline law removes only the common no-IP part of the IP error.  The
dominant Debye-IP near-source `Hz` residual remains the separate clean-delta
source-history/MMR problem and still needs a term that scales with
`delta_sigma` and vanishes for `delta_sigma = 0`.

This separation is now reproducible through
`atem3d.clean_delta_decomposition_cli`.  With the sign convention
`error = numerical - reference`, it computes

```text
ideal_clean_delta = noip_error - raw_ip_error
actual_correction = corrected_ip_numerical - raw_ip_numerical
correction_mismatch = actual_correction - ideal_clean_delta
corrected_ip_error = noip_error + correction_mismatch
```

The report

```text
outputs\hj_debye_ip_sub5_noip_law_clean_delta_decomposition.json
```

uses the raw IP sub5 report, the no-IP-law IP replay report, and the no-IP sub5
report over the three `Hz` receivers.  It gives
`actual_correction_relative_l2_to_ideal = 1.238/1.275/1.237`, with identity
residual below `1.1e-22`.  This confirms that the no-IP source-cell law is not an
IP clean-delta correction: the future production term should be judged by
reducing this clean-delta mismatch while leaving the no-IP baseline law as a
separate problem.
