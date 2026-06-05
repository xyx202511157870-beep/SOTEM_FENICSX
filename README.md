# ATEM3D: Grounded-Wire TDEM With Debye IP

This repository implements a direct time-domain finite-volume prototype for a
grounded electric wire source in 3D media, with induced-polarization memory
coupled directly into the time-stepping equations.

The intended environment is:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe
```

Install the project into that environment:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pip install -e .
```

## Numerical Formulation

The solver uses an EB finite-volume formulation on a `discretize.TensorMesh`.
Electric field lives on edges and magnetic flux density lives on faces:

```text
(b^{n+1} - b^n)/dt + C e^{n+1} = 0
C^T M_{mu^-1} b^{n+1} = j_c^{n+1} + s_e^{n+1}
```

The IP current is represented in conductivity Debye form:

```text
j_c = sigma_infinity e - sum_i delta_sigma_i y_i
tau_i dy_i/dt + y_i = e
```

Backward Euler gives:

```text
y_i^{n+1} = alpha_i y_i^n + beta_i e^{n+1}
alpha_i = tau_i/(tau_i + dt)
beta_i = dt/(tau_i + dt)
```

After eliminating the Debye variables, each time step solves:

```text
[C^T M_{mu^-1} C + (1/dt) M_{sigma_eff}] e^{n+1} = rhs
sigma_eff = sigma_infinity - sum_i beta_i delta_sigma_i
rhs = (1/dt) [M_{sigma_infinity} e^n
    - sum_i beta_i M_{delta_sigma_i} y_i^n
    - (s_e^{n+1} - s_e^n)]
```

For non-CPML runs this is the source-difference E-form used in the code.  When
the stored magnetic field satisfies the previous Ampere balance it is
equivalent to the magnetic-history form.  Active CPML still uses the
magnetic-history form because the split-curl memories are derived from the
stretched Faraday/Ampere equations.

The grounded wire is projected onto mesh edges using SimPEG's
`segmented_line_current_source_term`, with a nearest-edge fallback for small
examples where the exact projector rejects boundary-adjacent endpoints.

See [docs/formulation_and_validation.md](docs/formulation_and_validation.md) for
the full implementation contract: grounded-wire source terms, direct time-domain
Debye IP coupling, initial fields, coordinate conventions, boundary limitations,
and the empymod acceptance sequence.
The focused source-history/MMR derivation notes are in
[docs/source_history_mmr_derivation.md](docs/source_history_mmr_derivation.md);
they summarize the current equation-level constraints, the BE `g0/g1` history
basis, and the Ampere-residual candidates that have been ruled out.
A concise Chinese H/J Debye-IP formula note is available at
[docs/hj_debye_ip_formulation_zh.md](docs/hj_debye_ip_formulation_zh.md).
The EB and H/J runtimes also expose a disabled-by-default
`magnetic_recovery_source_history` block for evaluating one or more explicitly
supplied source-moment history coefficient sets against the FV/MMR receiver
matrix.  This is an integration hook for derived coefficients, not a fitted
production law.  The prescribed and driven source-history terms accept either
absolute `coefficients` or `normalized_coefficients`; the latter are scaled at
runtime by `mu * sum(delta_sigma_source) * source_length^2`, matching the matrix
audit convention while preserving the IP-dependent zero-contrast guard.  The
same block now also accepts the diagnostic kind
`time_series_source_moments`, which applies an explicit coefficient table on
the simulation time nodes.  It is intended for checking local-operator output
without forcing it into a single scalar history basis.  The block also accepts
`source_diffusion_kernel_source_moments`, which evaluates
`A exp(-(t-t0)/(m * mu * sigma_midpoint * L^2))` by default
(`basis_kind: continuous`) against the selected source-moment response.  The
same diagnostic can use `basis_kind: be_decay` to evaluate the
first-gate-normalized backward-Euler Debye decay on the simulation time grid
before applying the source-moment response.  `A` can be supplied as an absolute
coefficient or as `normalized_amplitude * mu * sigma_midpoint * L^2`, which lets
one dimensionless constant be replayed across meshes.  It is active even for
no-IP H/J runs and is only a runtime/postprocess replay of the current
source-neighborhood audit clue, not a derived production correction.  The block
metadata written by `empymod_validation` now records `requires_ip`; this field is
`false` only for this no-IP source-diffusion baseline diagnostic and `true` for
IP-only source-history candidates that must vanish when Debye terms are absent.
The block also accepts
`source_primary_delta6_source_moments`, a diagnostic bridge that expresses the
H/J `delta6` face-current source-primary test as a degree-zero FV face-source
moment.  It is useful for checking the runtime/postprocess path, not for
promoting delta6 into production physics.  The block also accepts
the non-fitted diagnostic kind
`initial_polarization_source_moments`, which projects the initial Debye
polarization current `-delta_sigma*y0` onto the selected source moments and
advances it with the BE relaxation kernel.  The source-centered H/J audit in
`outputs\initial_polarization_source_history_candidate_source_centered_eval.json`
rules out this simple candidate as the missing production term: its `Hz`
relative L2 against the IP residual is about `41`, with a candidate norm about
40 times too large.  The matrix-fit and matrix-scan diagnostics now support the H/J
face-current source space with `--field-location face` and
`--spatial-basis source_moments`, and report the static `B_R v_s` response rank
separately from the time-expanded least-squares design matrix; the first xy6
center-`Hz` audit is saved in
`outputs\source_history_matrix_scan_hj_face_current_reference_delta_hz_x0_source_moments_even02.json`.
The full three-point H/J Debye sampled report and IP-residual rank audit are
saved in
`outputs\hj_debye_ip_line_exeyhz_xy6pad_data_only_t010_1ms_samples_report.json`
and
`outputs\source_history_matrix_scan_hj_face_current_ip_residual_hz_line_source_moments_even02.json`;
the three-point `Hz` line gives static source-moment rank `2` for the `{0,2}`
face-moment basis, while the center-only scan is rank `1`.  The scan JSON also
contains `spatial_time_series`, a per-time projection of the target residual
onto the static source-moment space.  For the three-point H/J `Hz` IP residual,
that projection has relative L2 about `6.56e-4`; higher BE cascade fits reduce
the remaining time-kernel error only with rapidly worsening conditioning, as
recorded in
`outputs\source_history_matrix_scan_hj_face_current_ip_residual_hz_line_source_moments_even02_orders1to6.json`.
A same-path H/J `current_biot` diagnostic is available in
`examples\hj_debye_ip_line_exeyhz_xy6pad_current_biot.yaml`; its sampled report
shows `Hz` relative L2 about `0.74-0.81` before source-history correction.
The older fitted, diagnostic-only order-2 source-history coefficients in
`examples\hj_debye_ip_line_exeyhz_xy6pad_current_biot_diagnostic_source_history_order2.yaml`
were fit against the previous one-channel source fallback; after the weighted
fallback they reduce `Hz` only to about `0.17-0.20`.  A fresh weighted-source
matrix scan is saved in
`outputs\source_history_matrix_scan_hj_current_biot_weighted_face_source_ip_residual_hz_line_source_moments_even02_orders1to3.json`.
The side `Ex` error has been traced separately to H/J face-source transverse
placement: the weighted fallback reduces the side `Ex` relative L2 from about
`0.74` to about `0.278`, and the source-centered H/J mesh in
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot.yaml` reduces
`Ex` to about `0.088/0.071/0.088` at `x=-20/0/20 m` when the source uses
`face_projection: axis_aligned`.
The source-history diagnostics now include `--target delta_residual`, defined as
`(IP_ref - noIP_ref) - (IP_num - noIP_num)`.  This removes the no-IP H/J
baseline error before fitting an IP-only correction.  On the same
source-centered `current_biot` `Hz` line, the delta-residual source-moment scan
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_orders1to4.json`
has static `{0,2}` projection error about `2.4e-4`.  Its all-window BE orders
1-4 give relative L2 `0.194`, `0.142`, `0.090`, and `0.053`; orders 5-6 in
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_orders5to6_all.json`
give `0.0317` and `0.0202` with normalized condition numbers about `1.56e4`
and `1.02e5`.  This supersedes the earlier sign-changing `ip_residual` trace as
the clean target for deriving a non-fitted IP source-history/MMR coupling.
Those reports also include `spatial_time_series.history_basis_fits`; fitting the
static `{0,2}` coefficient traces directly gives the same story without the
receiver matrix in the second stage.  Order 1 has aggregate trace relative L2
about `0.193` with normalized coefficients
`[[6.758, 2.139], [3.415, 0.520]]`, while order 6 reaches about `0.0204`.
The remaining production gap is therefore the H/J time-kernel/coefficient law,
not the static source-moment projection.
The next non-fitted route is the H/J magnetic-diffusion recovery spectrum: the
small-support operator
`M_mu dh/dt + C.T M_rho C h = 0`, or equivalently
`C.T M_rho C h = lambda M_mu h`, gives local recovery rates that can be
coupled to Debye source memory without reading coefficients from empymod.  The
new `atem3d.recovery_spectrum` helpers compute these positive eigenmodes on
small diagnostic supports; they are a building block for the missing
MMR/source-recovery law, not yet the production correction itself.
The diagnostic `driven_recovery_source_moments` block accepts either a single
`response_tau` or multiple `response_taus`, so a future local-spectrum
derivation can inject several modal recovery times in one source-history term.
A reproducible local-spectrum report can be generated from a config and either
explicit local cells or a source/receiver-derived support.  For the H/J
grounded-wire diagnostics, the source/receiver path uses active face-source
dofs plus receiver-nearest cells:

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

The source-centered example report uses 210 local cells and 906 H/J edge dofs;
its first six homogeneous recovery time constants are `O(2e-8 s)`.  This is a
support/spectrum diagnostic, not yet the missing `O(0.1 ms)` source-history
kernel.
With `--include-modal-coupling`, the companion report
`outputs\local_recovery_spectrum_source_receiver_modal_coupling_demo.json`
projects the local face source moments through the same modes to the
`face_current_biot` receiver.  The first six modes remain far too fast; the
largest steady modal source/receiver factor is about `1.12e-2`, so this
homogeneous local support is a constraint on the next forcing/projection
derivation rather than a finished correction.  The report also records the
eigenvalue filter: hundreds of near-null modes are discarded, and the first
kept mode jumps to about `4.8e7 1/s`, leaving no local homogeneous mode near
the desired `1e4 1/s` (`0.1 ms`) scale.
The same modal report now projects each modal receiver response back onto the
source-moment response basis used by the runtime hook.  This modal-to-source
projection has aggregate receiver-space L2 `0.342`.  Collapsing the projection
with the compact `[8.1876, 2.0719]` normalized amplitude gives
`outputs\source_history_hj_rise_decay_kernel_prescribed_modal_projection_geometry_amp.json`,
with trace L2 `0.394` and summed normalized coefficients `[5.83, 4.01]`.
Thus the plain small-support homogeneous modal projection is not the missing
law, although the same projection path remains useful for larger or
nonhomogeneous local recovery operators.
The corresponding runtime/postprocess candidate is saved in
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_modal_projection_geometry_amp.yaml`.
Applied to the source-centered H/J data-only result, its empymod comparison
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_modal_projection_geometry_amp_t010_1ms_validation.json`
gives `Hz` relative L2 about `0.295/0.350/0.296`: better than the uncorrected
`0.588/0.650/0.589`, but worse than the trace-fitted geometry multi-tau
diagnostic `0.171/0.176/0.172`.
The recovery-spectrum CLI also has `--include-driven-response`, which solves
the local nonhomogeneous BE problem
`M_mu (h^{n+1}-h^n)/dt + C.T M_rho C h^{n+1}
= C.T M_rho s_f g^{n+1}` and projects the resulting receiver trace back to
source moments.  The report
`outputs\local_recovery_spectrum_source_receiver_driven_response_demo.json`
uses the saved `1 ms` Debye driver.  Collapsing its projected time-dependent
coefficients with `[8.1876, 2.0719]` gives trace L2 `0.217` against the H/J
`delta_residual` coefficient trace; the optimal scalar is `1.007`, so the
remaining error is mainly early-time shape rather than bulk amplitude.
Driven-only `--skip-spectrum` runs show that simply enlarging this local
support does not fix that shape: `sr0/rr0/pad0`, `sr1/rr1/pad0`,
`sr0/rr1/pad1`, and `sr1/rr1/pad1` all stay near trace L2 `0.216-0.218`.
Switching the driver from Debye decay to `--driver-kind debye_build_up` fixes
the first gate scale (`[0.77, 0.21]` versus target `[0.82, 0.15]`) but overshoots
late time (`[5.14, 1.42]` versus `[3.15, 0.79]`) and worsens trace L2 to about
`0.568`.  The missing kernel is therefore not a pure decay or monotone build-up;
it must be a rise-decay/constrained recovery response.
The same driven response can be generated with
`--driven-source-projection charge_conserving`; the real source-centered report
`outputs\local_recovery_spectrum_source_receiver_driven_response_charge_conserving_demo.json`
is numerically identical to the raw-source report to roundoff
(`max |Delta receiver| ~= 1.8e-13`).  This follows from the discrete identity
`C.T G phi = 0`: subtracting a gradient return-current projection before
`C.T M_rho s_f` cannot change this local magnetic forcing, so early-time error
must be fixed by a different constrained/global recovery operator.
The driven-response CLI now also exposes
`--driven-initial-state charge_conserving_mmr` and
`--driven-initial-state global_charge_conserving_mmr`.  The first builds a
nonzero local MMR initial `h` state from the charge-conserving source-moment
face currents; the second builds the same state on the full mesh and restricts
it to the local support using the support's global edge-dof map.
The source-centered audit
`outputs\local_recovery_spectrum_charge_conserving_mmr_initial_trace_audit.json`
shows that both nonzero initial states are indistinguishable from the
zero-initial driven response on the sampled `0.1-1 ms` window: all three give
trace L2 `0.2154` after fitting a constant source-drive amplitude, and all give
`0.2168` with the compact normalized amplitude `[8.1876, 2.0719]`.  The MMR
initial state decays on the very fast support diffusion scale, so it is a useful
diagnostic building block but not the missing production term; any global
return-current effect has to enter as an ongoing boundary/operator coupling,
not as a one-time initial condition.
The next forcing-level check is
`--driven-forcing global_mmr_steady`, which replaces the usual local
`C.T M_rho s` RHS by `K_local h_global`, where `h_global` is the full-mesh
charge-conserving MMR source-moment state restricted to the same support.  The
source-centered report
`outputs\local_recovery_spectrum_source_receiver_driven_response_global_mmr_steady_forcing.json`
and audit
`outputs\local_recovery_spectrum_global_mmr_steady_forcing_trace_audit.json`
again do not improve the missing trace: fitted-amplitude L2 remains `0.2154`,
and the compact-amplitude L2 is `0.2183`.  The first selected gate is still
about `[7.40, 2.13]` in normalized source-moment coordinates while the target
is `[0.82, 0.15]`.  Thus a steady global MMR forcing changes neither the
early-time scale nor the required rise-decay behavior.
The latest driven-only relaxation-difference scan drives the same local
magnetic-diffusion solve with `g_slow - g_fast`, using each support's
`mu*sigma*L^2` estimate as the fast time.  Collapsing the projected coefficients
with the compact normalized amplitude `[8.1876, 2.0719]` gives trace L2 values
`0.186`, `0.172`, and `0.321` for `sr0/rr0/pad0`, `sr1/rr1/pad0`, and
`sr1/rr1/pad1`, respectively; the best case is
`outputs\local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_sr1_rr1_pad0.json`
with `fast_tau = 5.6330878831955205e-5 s`, `210` cells, and `906` edge dofs.
An optimal scalar on that best collapsed trace is only `1.049` and lowers trace
L2 to `0.165`, so this is the strongest current non-fitted time-shape clue.
It still uses the compact amplitude inferred from coefficient sums, so it
remains diagnostic-only until that amplitude is derived from FV/MMR operators.
Combining the same `sr1/rr1/pad0` rise-decay driver with
`--driven-forcing global_mmr_steady` in
`outputs\local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_global_mmr_steady_forcing_sr1_rr1_pad0.json`
does not materially improve the trace.  The audit
`outputs\local_recovery_spectrum_relaxdiff_global_mmr_steady_forcing_trace_audit.json`
gives fitted-amplitude L2 `0.1637` for both the ordinary source RHS and the
global-MMR steady RHS, while the compact-amplitude L2 is slightly worse for the
global forcing (`0.1736` versus `0.1717`).  This keeps the focus on deriving the
dynamic rise-decay kernel and amplitude law rather than changing only the steady
spatial RHS.
The corresponding runtime/postprocess smoke configs are
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp.yaml`
and its `_scaled` variant.  Their empymod validation reports give `Hz` relative
L2 about `0.236/0.250/0.236` and `0.222/0.236/0.223`, respectively.  This
postprocess result is useful but weaker than the older one-mode driven hook
(`0.194/0.205/0.195`), showing that the local driven-response matrix's
time-dependent projection is not captured by a single static
`response_tau + coefficients` term.
The fitted two-component amplitude from the same trace audit was also replayed
through
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_geometry_amp_fitted.yaml`.
It gives `Hz` L2 `0.221/0.237/0.222`, so replacing the compact amplitude by the
best constant trace amplitude does not close the runtime gap.
The same sr1/rr1/pad0 local driven-response projection was then applied through
the new explicit `time_series_source_moments` hook in
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_relaxdiff_time_series_geometry_amp.yaml`
and its `_scaled` variant.  This is a stricter replay of the local projected
coefficient matrix, but the empymod validation gives `Hz` relative L2 about
`0.251/0.267/0.252` and `0.238/0.253/0.239`, respectively.  Thus the local
projection report is still only a trace-space/operator clue; it is not yet a
runtime source-history law.  A direct receiver-space replay of the same local
`receiver_response` gives the same numbers, and a negative-amplitude control
worsens `Hz` to about `0.973/1.097/0.974`; the sign is not the issue.  The
remaining error is in the local source/return-current recovery operator or its
non-fitted amplitude, not in the replay mechanism.
Padding the same source/receiver support to the full simulation mesh gives a
dynamic full-domain driven-response check without a dense spectrum solve:
`outputs\local_recovery_spectrum_source_receiver_driven_response_relaxation_difference_fullpad_global_dynamic.json`
and
`outputs\local_recovery_spectrum_relaxdiff_fullpad_global_dynamic_trace_audit.json`.
These driven-response trace audits are reproducible with
`atem3d.recovery_driven_trace_audit_cli`.
With the geometry fast time `5.633e-5 s`, the fitted trace L2 is `0.1628`, only
slightly different from the local `0.1637`, and the compact-amplitude L2
worsens to `0.190`.  With empirical `fast_tau=0.11 ms`, the full-domain report
`outputs\local_recovery_spectrum_relaxdiff_fullpad_empirical_tau_trace_audit.json`
recovers the expected single-tau clue (`0.1215` fitted trace L2), but the first
fitted gate is still `[4.51, 1.30]` versus the target `[0.82, 0.15]`.  Enlarging
the support to the full domain is therefore not the missing production law by
itself.
The sweep report
`outputs\local_recovery_spectrum_source_receiver_sweep.json` varies source
cell radius, receiver cell radius, and padding.  Its geometric
`mu*sigma*L^2` estimates span `5.6e-5` to `2.0e-4 s`, which covers the empirical
`0.1 ms` scale, but every computable homogeneous spectrum still keeps only
`O(1e7 1/s)` modes after the near-null block.  The slow scale is therefore a
source/receiver recovery-path effect to derive, not a plain local curl-curl
eigenvalue.
The trace-kernel bridge
`outputs\source_history_hj_rise_decay_kernel_from_recovery_sweep_tau_estimates.json`
feeds those sweep `tau_est` values into the saved H/J `delta_residual`
coefficient traces.  The best single geometry tau is `5.63e-5 s` with relative
L2 `0.164`, worse than the empirical all-window `0.11 ms` fit (`0.122`), while
the three unique geometry-tau basis functions can fit to about `0.039` with a
column-normalized condition number around `251`.  This keeps the geometry
scale as useful evidence, but not as a finished coefficient law.
The same trace-kernel report now records coefficient sums over the history
basis.  For the three geometry-tau fit, the summed normalized source-moment
coefficients are about `[8.19, 2.07]` in units of `mu*delta_sigma*L^2`.  This
near-collapses the fitted multi-tau table back to a compact source-moment
amplitude, suggesting that the remaining derivation may be a rule for
distributing this amplitude across recovery paths rather than for inventing a
new static source vector.
The trace-kernel CLI can now evaluate such prescribed physical or normalized
coefficients directly.  Concentrating the summed normalized amplitude
`[8.1876, 2.0719]` on the best geometry tau `5.633e-5 s` gives trace-space
relative L2 `0.173`; rounding it to `[8, 2]` gives `0.181`.  Using the same
amplitude at the empirical `0.11 ms` tau gives `0.179`, even though a fitted
single coefficient at that tau reaches `0.122`.  Thus the compact amplitude is
a useful physical clue, but the non-fitted law still has to determine the
recovery-path time distribution and not just the total source moment.
The latest non-fitted distribution check
`outputs\source_history_hj_driven_relaxation_geometry_tau_moment_constraint_summary.json`
tests geometry-tau weights constrained by amplitude preservation plus early
moment cancellation.  The best of those prescribed candidates gives trace L2
about `0.186`, far from the fitted three-geometry-tau value `0.039`, so the
modal distribution is not explained by simple interpolation or low-order
initial-slope cancellation.
The companion shell-weight check
`outputs\source_history_hj_driven_relaxation_geometry_tau_shell_weight_summary.json`
finds a small signal in the nested-support pattern `[-1, 2, 0]`: it improves a
driven single-geometry-tau `[8,2]` baseline from L2 `0.200` to `0.181`, while
other simple shell patterns are worse.  This points back to a real FV/HJ
source-to-recovery projection over support shells, not to hand-picked
inclusion-exclusion weights.
The same check has now been repeated on the actual projected driven-response
reports rather than on idealized tau columns.  The CLI
`atem3d.recovery_driven_trace_audit_cli` accepts `--combine-report-weights`,
and the summary
`outputs\local_recovery_spectrum_relaxdiff_actual_driven_shell_trace_audit_summary.json`
records three combinations of the `sr0/rr0/pad0`, `sr1/rr1/pad0`, and
`sr1/rr1/pad1` reports.  The actual `[-1, 2, 0]` shell gives fitted trace L2
`0.16364` and compact `[8,2]` L2 `0.16497`, with fitted normalized amplitude
`[7.995, 1.829]`.  The alternatives `[-1, 3, -1]` and `[1, -2, 2]` are much
worse (`0.28280`/`0.40580` and `0.41793`/`0.51295` for fitted/compact L2).
Thus the actual FV projection strengthens the shell clue, but the first selected
gate is still too large (`[5.19, 1.39]` compact versus target
`[1.34, -0.22]` in `mu*delta_sigma*L^2` units).  The missing law is not a
simple support-shell finite difference; it must supply the early-time
suppression and the signed moment-2 response from the source/MMR recovery
operator.
As a diagnostic target for that non-fitted law, the same CLI can also fit
report weights with the compact amplitude held fixed via
`--fit-compact-report-weights`.  On the three actual reports above, fixed
`[8,2]` amplitudes reach trace L2 `0.08332` with report weights
`[-6.666, 7.627, 0.131]` and condition number about `197`.  This is much better
than hand-picked shell weights, but it relies on large cancellation between the
two smallest supports and still leaves the first gate with the wrong moment-2
sign (`[2.53, 0.99]` versus `[1.34, -0.22]`).  The next production derivation
therefore needs a finite-volume source/MMR weighting operator whose spectrum can
produce this cancellation without fitting empymod residuals.
Rounding those fitted report weights in shell coordinates gives a compact
integer candidate: independent shell weights `[1, 8, 0]`, equivalent to report
weights `[-7, 8, 0]`.  Its trace audit
`outputs\local_recovery_spectrum_relaxdiff_actual_driven_shell_mid8_trace_audit.json`
is strong: compact `[8,2]` L2 `0.10631`, compact after one scalar `0.08426`,
and two-amplitude fitted L2 `0.05291` with normalized amplitude
`[8.726, 1.608]`.  However the sub5 runtime/time-series replay summarized in
`outputs\hj_relaxdiff_actual_shell_mid8_runtime_summary.json` only reaches
`Hz` relative L2 about `0.228/0.250/0.228` for compact amplitudes and
`0.229/0.252/0.230` for fitted amplitudes.  So `[1,8,0]` is a useful
support-weight clue, but it does not close the receiver-space problem; the
missing production term still has to derive the source/MMR recovery-to-receiver
operator, not only a source-moment trace.
The follow-up decomposition now comes from
`atem3d.clean_delta_decomposition_cli`, which shows why the runtime error does
not fall to the trace residual.  With the sign convention
`error = numerical - reference`, the ideal IP-only correction is
`noip_error - raw_ip_error`; the final corrected error is
`noip_error + correction_mismatch`.  For the sub5 `[-7,8,0]` compact replay,
`outputs\hj_debye_ip_sub5_relaxdiff_actual_shell_mid8_compact_clean_delta_decomposition.json`
records correction-mismatch L2 about `0.026/0.037/0.026` and aggregate
`actual_correction_relative_l2_to_ideal = 0.0759`.  The trace-fitted-amplitude
variant
`outputs\hj_debye_ip_sub5_relaxdiff_actual_shell_mid8_fitted_amp_clean_delta_decomposition.json`
is cleaner in IP-only space, with mismatch `0.0169/0.0157/0.0171` and aggregate
`actual_correction_relative_l2_to_ideal = 0.0421`, even though its final
receiver-space `Hz` L2 is slightly worse than the compact replay.  The older
one-mode driven-recovery replay
`outputs\hj_debye_ip_sub5_driven_recovery_clean_delta_decomposition.json`
has larger mismatch `0.063/0.072/0.063`, while the no-IP source-cell law is not
a clean-delta term at all (`actual_correction_relative_l2_to_ideal` about
`1.24/1.27/1.24`).  The no-IP/current-Biot baseline is already about
`0.214/0.239/0.214` when measured against the IP reference norm, so the final
`0.228/0.250/0.228` compact runtime errors are mostly the no-IP magnetic-recovery
baseline plus a small constructive mismatch.  This separates the remaining work:
the Debye/IP source-history branch is now close to a clean-delta correction,
while complete empymod agreement still needs the no-IP H/J magnetic recovery
baseline reduced.
The no-IP baseline is now further localized with
`atem3d.time_warp_audit_cli`, which fits
`reference_time = scale * numerical_time + shift` directly on validation-report
samples.  For the source-centered sub5 H/J `current_biot` no-IP report,
`outputs\hj_noip_source_centered_current_biot_sub5_time_warp_audit.json` shows
that the three `Hz` L2 values `0.174/0.181/0.174` collapse to about
`0.0085/0.0086/0.0085` under a common affine map with scale `1.1175` and shift
about `-32.5` to `-33 us`.  The old xy6 H/J signfix audit
`outputs\hj_noip_xy6_signfix_time_warp_audit.json` similarly reduces
`0.184-0.186` to about `0.014` with scale about `1.42-1.43`.  By contrast, the
EB/current-Biot no-IP companion
`outputs\eb_noip_current_biot_sub5_time_warp_audit.json` only needs a mild
scale near `0.963-0.965` and reaches about `0.004`.  This makes the remaining
H/J no-IP issue look like a face-source/MMR diffusion-time-scale defect, not a
simple magnetic receiver amplitude, sign, or absorbing-boundary error.
A targeted H/J z-refinement check is saved in
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_zrefined_current_biot_sub5.yaml`.
It splits the top 1 m source cell into `0.25/0.5/0.25 m` while keeping the
source/receiver at the center of the 0.5 m source cell.  Its report
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_zrefined_current_biot_sub5_t010_1ms_samples_report.json`
worsens raw `Hz` L2 to about `0.196/0.202/0.195`, but the time-warp audit
`outputs\hj_noip_source_centered_zrefined_current_biot_sub5_time_warp_audit.json`
reduces the required scale from `1.1175` to about `1.0225`.  The first and last
`Hz` ratios are both high (`~1.31` early and `~1.10` late), so z-refinement
removes most of the diffusion-time-scale bias while exposing a separate
near-source magnetic amplitude/spatial-kernel error.
Two postprocessed spatial-shell audits decompose the sub5 `current_biot` `Hz`
receiver response by cell distance from the source line.  On the original mesh,
`outputs\hj_noip_source_centered_current_biot_sub5_spatial_shell_fit.json`
reconstructs the report numerical data to machine precision, then shows that a
single scalar brings combined `Hz` L2 from `0.176` to `0.103`, while seven free
radial shell weights only reach `0.088` and require large alternating signs.
On the z-refined mesh,
`outputs\hj_noip_source_centered_zrefined_current_biot_sub5_spatial_shell_fit.json`
gives raw combined L2 `0.198`, a single scalar L2 `0.079`, and a shell fit
`0.087`, again with unstable alternating shell weights.  Thus the near-source
problem is not a clean "one radial shell is wrong" error; the next non-fitted
route should derive a source-cell/receiver magnetic kernel or MMR transfer law
rather than fitting empirical shell weights.
A direct source-channel replacement was also tested with the no-IP full fields:
`outputs\hj_noip_source_channel_line_kernel_replacement_audit.json` and
`outputs\hj_noip_source_channel_line_kernel_replacement_zrefined_audit.json`
project `curl H` onto the active H/J source faces and replace that component's
discrete face-current Biot response by the continuous finite-wire Biot response.
The positive-time source-channel amplitudes are only `O(1e-6)` after the skipped
early gates, and the replacement improves side `Hz` L2 by only about `0.001-0.002`
while leaving the center almost unchanged.  Therefore the remaining error is not
the active source-channel magnetic kernel by itself; it lives in the broader
source-neighborhood MMR/diffusive current transfer.
A stricter source-channel cell-current line-segment replacement reaches the
same conclusion.  The reports
`outputs\hj_noip_source_channel_cell_line_segment_kernel_audit.json` and
`outputs\hj_noip_source_channel_cell_line_segment_kernel_zrefined_audit.json`
replace the reconstructed source-channel cell `Jx` contribution by finite line
segments.  On the original mesh the three `Hz` L2 values only move from about
`0.17367/0.18064/0.17384` to `0.17335/0.18032/0.17351`; on the z-refined mesh
they move from `0.19615/0.20221/0.19464` to `0.19596/0.20202/0.19445`.  The
active/source-channel current itself is therefore too small to be the missing
near-source magnetic term.
The broader face-current neighborhood was then audited with
`atem3d-source-neighborhood-audit`, which groups `C h - s_face` by radial
face-current shells around the finite source segment and keeps the same
`current_biot` receiver matrix.  The original report
`outputs\hj_noip_source_centered_current_biot_sub5_source_neighborhood_audit.json`
reconstructs the numerical samples to machine precision and shows that fitting
the `0-80 m` source-neighborhood shells while keeping the `>=80 m` outside fixed
can reduce combined `Hz` L2 from `0.176` to `0.0091`.  However the shell weights
are large and alternating
`[-46.4, 2.18, -3.89, 28.2, -29.5, 6.17]`, with condition number about
`4.55e3`.  The z-refined companion
`outputs\hj_noip_source_centered_zrefined_current_biot_sub5_source_neighborhood_audit.json`
shows the same pattern: `0.198` to `0.0095`, weights
`[-40.4, 10.8, -20.9, 75.4, -37.5, 5.15]`, and condition number about
`4.43e3`.  This localizes the no-IP `Hz` residual to the source-neighborhood
transfer subspace, but it also rules out promoting empirical radial weights into
production physics; the next step must derive the FV/HJ source/MMR weighting and
diffusion kernel that produces this cancellation.
A matrix-free H/J diagnostic receiver mode, `face_basis_biot`, now applies the
same `C h - s_face` current through the face-basis Biot integral instead of the
cell-averaged `face_current_biot` matrix.  A representative three-gate replay on
the `sourcecell0p25` and `sourcecell0p75` sub5 no-IP reports worsens aggregate
`Hz` L2 from about `0.296` to `0.336`, so the remaining no-IP baseline is not
fixed by replacing the receiver quadrature alone.
The same audit now projects non-fitted static face-current candidates against
the residual.  The raw H/J source face vector (`active_source`) and its
charge-conserving projection have excellent per-time spatial alignment with the
missing `Hz` correction: on the original mesh their per-time scalar projection
errors are about `0.0117` and `0.0123`; on the z-refined mesh they are about
`0.0135` each.  A single all-window scalar still leaves L2 about
`0.173-0.178`, so the missing law is not a constant source-primary amplitude;
it is a time-dependent source coefficient.  The fitted coefficient trace decays
by a factor about `0.778` in the first `10 us` on the original mesh and `0.792`
on the z-refined mesh, corresponding to a first-step exponential time scale of
roughly `40-43 us`, comparable to `mu0 * sigma * L^2 ~= 31 us` for a `50 m`
source in the `0.01 S/m` layer.  The DC initial current candidate is much worse
spatially (`~0.166/0.186` per-time L2 for original/z-refined), so the next
derivation should focus on a source-face/MMR diffusion-time kernel, not on the
long-on-time DC current shape.
The candidate audit also records exponential coefficient-kernel checks at fixed
multiples of that non-fitted source diffusion time.  For the original mesh,
`active_source` with `tau = 1.25 * mu * sigma * L^2` reduces the corrected
combined `Hz` L2 to `0.0231` after fitting only the scalar amplitude; the
coefficient-trace L2 is `0.113`.  On the z-refined mesh, the best checked value
is `tau = 1.5 * mu * sigma * L^2`, giving corrected L2 `0.0286` and
coefficient-trace L2 `0.127`.  The first-gate amplitude gives almost the same
coefficient-trace error as the fitted amplitude in both cases, which makes this
the strongest current no-IP H/J source-kernel clue.  It is still not production
physics because the amplitude and the `1.25-1.5` geometry factor have not yet
been derived from the discrete H/J source update.
That source-diffusion kernel is now available as the diagnostic-only
`source_diffusion_kernel_source_moments` runtime/postprocess term.  The original
sub5 replay config
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_kernel.yaml`
uses amplitude `-1.106396079306176e-4`, `tau_multiplier: 1.25`, and
`amplitude_time: 1.0e-4`; postprocessing
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_data_only.h5`
writes
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_kernel_data_only.h5`.
The empymod comparison
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_kernel_t010_1ms_validation.json`
gives `Hz` relative L2 about `0.0219/0.0248/0.0224` over `0.1-1 ms`.
The z-refined replay config
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_zrefined_current_biot_sub5_diagnostic_source_diffusion_kernel.yaml`
uses amplitude `-1.153342159870617e-4` and `tau_multiplier: 1.5`; its report
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_zrefined_current_biot_sub5_diagnostic_source_diffusion_kernel_t010_1ms_validation.json`
gives `Hz` relative L2 about `0.0284/0.0308/0.0263`.  This confirms the
runtime/postprocess path, but the term remains diagnostic because its amplitude
and geometry multiplier are still fitted from the audit rather than derived.
A cross-report coefficient-law audit is saved in
`outputs\hj_noip_source_centered_source_diffusion_law_audit.json`.  It reads the
two source-neighborhood reports and compares the normalized coefficient traces.
The per-report best amplitudes normalize to
`A/(mu sigma L^2) = -3.522` and `-3.671`, with mean `-3.596` and coefficient of
variation about `2.1%`; the first-gate coefficients normalize to `-3.485` and
`-3.713`, with mean `-3.599` and coefficient of variation about `3.2%`.
The best single checked global law over the two reports is
`tau_multiplier = 1.5`, `A/(mu sigma L^2) = -3.468`, with combined coefficient
trace L2 `0.148`.  This makes the amplitude normalization a stronger candidate
for derivation than the current mesh-dependent time multiplier, but it still
does not prove a production formula.
The same global law is now replayable directly through the runtime config field
`normalized_amplitude`.  The paired configs
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_global_law.yaml`
and
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_zrefined_current_biot_sub5_diagnostic_source_diffusion_global_law.yaml`
both use `normalized_amplitude: -3.467727352856916` and
`tau_multiplier: 1.5`.  Postprocessed empymod comparisons give original-grid
`Hz` L2 about `0.0256/0.0373/0.0262` and z-refined `Hz` L2 about
`0.0337/0.0259/0.0316`.  This loses some per-grid fitted accuracy but is a
stricter cross-grid check because the absolute amplitude is now computed from
`mu sigma L^2` inside the runtime.
The companion geometry audit CLI records the source-vector metrics needed to
continue deriving that normalization rather than fitting it:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_geometry_audit_cli `
  examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_global_law.yaml `
  -o outputs\hj_noip_source_centered_current_biot_sub5_source_geometry_audit.json
```

It reports the runtime source-vector sign/location, active H/J face counts,
orientation counts, face-area weighted norms, source midpoint cell widths,
midpoint conductivity, `tau0 = mu sigma L^2`, and the configured
`source_diffusion_kernel_source_moments` amplitude or normalized amplitude.
For H/J `current_biot` configs it also reports the receiver-side static
response `B_R v_0`.
The first generated reports
`outputs\hj_noip_source_centered_current_biot_sub5_source_geometry_audit.json`
and
`outputs\hj_noip_source_centered_zrefined_current_biot_sub5_source_geometry_audit.json`
show the same `tau0 = 3.141592653175e-5 s` and 11 active x-faces.  Raw source
`L1/L2` doubles under z-refinement (`2.2/0.6633` to `4.4/1.3266`), while the
area-weighted `L1` stays `11.0`.  The H/J inner-product/RHS norms are not
invariant (`||C^T M_rho v_0||` grows from `2.39e3` to `4.71e3`), but the
receiver-projected source shape is: `B_R v_0` has `Hz` values about
`[0.012715, 0.015258, 0.012715]` and
`[0.012725, 0.015269, 0.012725]`.  This points the next derivation toward the
composite receiver-projected source-neighborhood transfer, not a raw source or
RHS norm.
The source-cell-thickness sweep
`outputs\validation_sweep_hj_noip_source_cell_thickness_t010_1ms.json`, driven by
`examples\validation_sweep_hj_noip_source_cell_thickness.yaml`, checks source
cell thicknesses `1.0/0.75/0.5/0.25 m` while keeping the source at `z=-0.5`.
Raw `Hz@x=0` L2 is about `0.181/0.186/0.202/0.185`, so this is not a monotone
mesh-convergence route; changing the H/J source cell trades diffusion-time
scale against near-source amplitude rather than solving both.
The matching geometry sweep
`outputs\validation_sweep_hj_noip_source_cell_thickness_source_geometry_audit.json`
shows why this is a dynamic, not static, source-shape issue: over source-cell
thicknesses `1.0 -> 0.25 m`, `||C^T M_rho v_0||` grows from `2.39e3` to
`9.39e3`, but `||B_R v_0||` stays near `2.36e-2`.  The unresolved
`tau_multiplier` must therefore come from the time-dependent FV/MMR
source-neighborhood transfer before receiver projection.
Two additional full-field source-cell cases now fill out that sweep:
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p75_current_biot_sub5.yaml`
and
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_sourcecell0p25_current_biot_sub5.yaml`.
Their source-neighborhood audits
`outputs\hj_noip_source_centered_sourcecell0p75_current_biot_sub5_source_neighborhood_audit.json`
and
`outputs\hj_noip_source_centered_sourcecell0p25_current_biot_sub5_source_neighborhood_audit.json`
reduce combined `Hz` L2 from `0.1809 -> 0.0120` and `0.1801 -> 0.00590`,
respectively.  The four-report source-cell law audit
`outputs\hj_noip_source_centered_source_cell_thickness_source_diffusion_law_audit.json`
gives normalized amplitude mean `A/(mu sigma L^2) = -3.604` with CV about
`1.48%`; the best single global law is `tau_multiplier = 1.25`,
`A/(mu sigma L^2) = -3.667`, coefficient L2 `0.132`, with
`tau_multiplier = 1.5` close behind at `0.138`.  This strengthens the
amplitude law and leaves the time multiplier as the weaker dynamic-FV/MMR
unknown.
The direction-audit reruns
`outputs\hj_noip_source_centered_sourcecell0p25_current_biot_sub5_source_neighborhood_direction_audit.json`
and
`outputs\hj_noip_source_centered_sourcecell0p75_current_biot_sub5_source_neighborhood_direction_audit.json`
now report residual-space projection metrics for the `active_source` candidate.
For source cells `0.25/0.75 m`, an arbitrary per-time scalar on the same
receiver-projected source vector explains `99.56%/99.56%` of the residual energy
(`per_time_residual_relative_l2 = 0.0665/0.0665`), while one all-window scalar
explains only `5.42%/3.53%`.  The missing no-IP H/J term is therefore mainly a
time-kernel/sign-evolution law on the grounded-source channel, not a different
static receiver-space source direction.
The z-refined direction rerun
`outputs\hj_noip_source_centered_zrefined_current_biot_sub5_source_neighborhood_direction_audit.json`
confirms the same structure (`per_time_residual_projection_fraction = 0.9953`).
The three-report BE source-diffusion law audit
`outputs\hj_noip_source_centered_direction_source_diffusion_law_audit_be_decay.json`
keeps the best global law at `tau_multiplier = 1.25`,
`A/(mu sigma L^2) = -3.556`, with coefficient L2 `0.126`.  Its direction
constraint summary gives per-time projection min/mean/max
`0.9953/0.9955/0.9956`, but all-window projection mean only `0.093`; this makes
the next derivation target specifically the discrete H/J source-channel time
response.
The same report now also estimates an early-gate effective BE decay multiplier
from adjacent coefficient ratios, using only same-sign decays whose starting
amplitude exceeds `5%` of the peak.  The weighted BE multiplier is
`1.168/1.166/1.312` for sourcecell `0.25/0.75/zrefined`, with mean `1.215`.
This independent ratio estimate is close to the global fitted `1.25`, so the
working non-fitted target is a BE-like source-channel decay on the scale
`tau_s approx 1.2-1.3 mu sigma L^2`, with the z-refined case indicating a
slightly broader recovery-time distribution.
The law audit also writes a copyable diagnostic source-history block.  The
matching replay config
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_direction_law_be_decay.yaml`
uses `normalized_amplitude = -3.555589582057861`,
`tau_multiplier = 1.25`, and `basis_kind = be_decay`; its empymod validation
report
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_direction_law_be_decay_t010_1ms_validation.json`
gives `Hz` L2 `0.0223/0.0344/0.0229` at `x=-20/0/20 m`.  This is the same
order as, but not uniformly better than, the older four-report BE replay
(`0.0224/0.0323/0.0230`), so it remains a reproducibility bridge rather than a
production correction.
The same four-report law was replayed through receiver postprocess using
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_*_diagnostic_source_diffusion_source_cell_law.yaml`
configs.  Empymod `Hz` L2 over `0.1-1 ms` is:
`1.00 m = 0.020/0.030/0.021`, `0.75 m = 0.023/0.029/0.024`,
`0.50 m = 0.040/0.033/0.038`, and `0.25 m = 0.015/0.023/0.016`.
So the law transfers to receiver data across the source-cell-thickness sweep,
but remains diagnostic because `a0` and `tau_multiplier` are still constrained
from residuals rather than derived from the FV/H/J operator.
`atem3d.recovery_driven_trace_audit_cli` now also accepts
`--target-source-neighborhood-report`, so no-IP active-source traces can be
audited against non-fitted local driven-response reports using
`source_diffusion_time_s = mu sigma L^2` normalization.  A degree-zero
`debye_decay` driven response on the 210-cell support gives fitted trace L2
`0.121/0.134/0.189/0.0968` over the `1.00/0.75/0.50/0.25 m` source-cell sweep,
but needs fitted amplitudes around `-57` to `-64` in `tau0` units.  Thus it is a
real time-shape clue, not yet the derived no-IP baseline correction.
The companion amplitude-scale audit
`outputs\local_recovery_noip_source_centered_source_cell_m1_amplitude_scale_audit.json`
compares that fitted bridge with source geometry and with the first selected
driven-response gate.  The apparent geometry match
`(pi/2)*(L/dx) = 15.708` is less direct than the BE time-basis explanation:
with `tau0 = pi*dt` and the first target at step 10,
`1/g0 = (1 + dt/tau0)^10 = 15.855`, matching the compact optimal scalars to
within the same residual pattern.  The driven-trace audit now also records the
first-gate-only compact scalar: it reduces compact trace L2 from about `0.94` to
`0.108-0.204`, close to the fitted-amplitude range `0.097-0.189`, with mean
`first_gate_scalar/(1/first_response) = 0.974`.  The current no-IP conclusion is
therefore a time-origin/first-gate normalization issue first, with a remaining
shape residual that must be derived from the FV/H/J source-neighborhood transfer
before interpreting any source-geometry factor.  The new per-time error-fraction
fields show that this residual is distributed over the early-middle gates
(`0.14-0.15 ms` peak for the source-cell sweep), and the 0.50 m z-refined case
remains the largest shape-mismatch outlier.  A targeted 0.50 m driven-response
`m` sweep shows that this outlier is mostly a recovery-time issue: changing
`driver_tau` from `1.0 tau0` to `1.25-1.5 tau0` reduces fitted trace L2 from
`0.189` to `0.136-0.129`, but the required scalar changes from `17.4` to
`9.9-6.6`, so this is a diagnostic constraint on the FV/H/J recovery-time
distribution, not a production amplitude law.  Completing the same `m` sweep for
all four source-cell targets shows the best compromise is `m = 1.25`
(mean/max fitted L2 `0.1185/0.1357`), while the per-case best `m` still drifts
from `1.0` to `1.5`; amplitude and recovery time therefore have to be derived
together from the discrete source-neighborhood operator.  A positive-spectrum
check on the same local supports shows the slowest kept local eigen-times are
only `2.1e-8` to `7.3e-7 s`, which is `43-1508` times faster than `tau0`; the
`m*tau0` scale is therefore not the bare slowest local eigenmode, but a
grounded-source driven-transfer timescale.  Padding the 0.50 m local support
from `210` to `1980` cells leaves the normalized fitted L2 unchanged
(`0.135681` at `m=1.25`, `0.128507` at `m=1.5`) and changes the scalar by only
about one percent, so the timescale is not a simple small-box boundary artifact.
The current source-projection/forcing toggles also do not explain it: on the
0.50 m, `m=1.25` audit, `charge_conserving`,
`charge_conserving_mmr_initial`, and `global_charge_conserving_mmr_initial`
match the raw selected target-window coefficients to roundoff, while
`global_mmr_steady_forcing` changes them by only relative L2 `6.34e-4`.
The driven-trace audit now also records driver-following error; for the 0.50 m
`m` sweep, selected source-moment traces match the imposed `driver_values` shape
to `1e-15`.  Thus the local driven solve is quasi-static over `0.1-1.0 ms`: it
provides a static transfer amplitude, while the time shape comes from the
externally prescribed history kernel.  The missing production derivation has
therefore moved to the grounded-source history kernel itself.  The
source-diffusion law audit now supports `--basis-kind be_decay`, a
first-gate-normalized BE Debye kernel matching the driven reports.  On the four
source-cell targets it keeps the global best `m=1.25` but changes the best
global amplitude/L2 from the continuous-basis `-3.6668/0.1318` to
`-3.5070/0.1286`, and the per-case BE best fits match the driven-response
`m` sweep.
The matching runtime diagnostic config
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_source_cell_law_be_decay.yaml`
sets `basis_kind: be_decay`; its validation report
`outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_diagnostic_source_diffusion_source_cell_law_be_decay_t010_1ms_validation.json`
records `Hz` relative L2 `0.0224/0.0323/0.0230` over `0.1-1 ms`.
Applied by itself to the Debye-IP sub5 H/J `current_biot` result through
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_noip_source_cell_law.yaml`,
the same no-IP law only moves IP `Hz` L2 from `0.414/0.476/0.414` to
`0.395/0.466/0.396`.  This confirms the separation: the no-IP baseline
correction is useful, but the dominant IP error still requires a distinct
clean-delta source-history/MMR law that vanishes with `delta_sigma`.
The new audit CLI

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.clean_delta_decomposition_cli `
  outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_t010_1ms_samples_report.json `
  outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_noip_source_cell_law_t010_1ms_validation.json `
  outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_t010_1ms_samples_report.json `
  --component-prefix Hz `
  -o outputs\hj_debye_ip_sub5_noip_law_clean_delta_decomposition.json
```

makes that separation explicit with the sign convention
`error = numerical - reference`.  It reports the ideal IP-only correction
`noip_error - raw_ip_error`, the actual candidate correction
`corrected_ip_numerical - raw_ip_numerical`, and their clean-delta mismatch.
For this no-IP-law replay, `actual_correction_relative_l2_to_ideal` is about
`1.24/1.27/1.24` for the three `Hz` receivers, while the no-IP baseline error
itself is only `0.214/0.239/0.214` against the IP reference norm.  This turns the
previous qualitative statement into a repeatable acceptance check: a production
IP source-history term must reduce the clean-delta mismatch, not merely replay
the no-IP source-neighborhood baseline.
The runtime guard now applies the same invariant: source-history corrections
whose metadata says `requires_ip: true` are skipped not only when the model has
no Debye terms, but also when all Debye `delta_sigma` arrays are zero.  The
diagnostic no-IP `source_diffusion_kernel_source_moments` term remains the only
`requires_ip: false` exception.
A diagnostic-only runtime/postprocess config,
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_geometry_tau_multimode.yaml`,
uses those three geometry-derived response times with the fitted trace
coefficients.  Applying it to
`outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_data_only.h5`
produces
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_geometry_tau_multimode_data_only.h5`;
the empymod report
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_geometry_tau_multimode_t010_1ms_validation.json`
reduces the three `Hz` relative L2 values to about
`0.171/0.176/0.172`.  This is a useful upper-bound runtime check for the
multi-`response_taus` path, but the coefficients are still trace-fitted.

Prescribed coefficients are now evaluated in that same trace space under
`spatial_time_series.prescribed_history_basis`; the round-trip order-1 audit in
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_source_moments_even02_order1_trace_prescribed.json`
records receiver-space prescribed L2 `0.19370` and trace-space prescribed L2
`0.19317`.  This is the hook for checking future non-fitted FV/HJ-derived
coefficient tables without confusing receiver-space and trace-space errors.
Both matrix CLIs now also accept
`--prescribed-normalized-coefficients`, interpreted in units of
`mu * delta_sigma * source_length^2`, so dimensionless tables from a derivation
can be audited directly.  Runtime `magnetic_recovery_source_history` configs
now accept the same convention as `normalized_coefficients` for prescribed and
driven source-moment terms.  The trace-kernel CLI accepts the same normalized
form for prescribed rise-decay coefficients, which is how the geometry-sum
candidate above was checked.  The rounded material-source candidate
`[[6, 2], [3, 0.5]]` is saved in
`outputs\source_history_matrix_hj_delta_residual_material_source_6_2_3_05_normalized_candidate.json`:
with the current source-centered sub5 sampled reports it gives receiver-space
prescribed L2 `0.198` and trace-space L2 `0.204`, close to but still above the
order-1 fitted bounds `0.183` and `0.182`.  Its refreshed window scan
`outputs\source_history_matrix_scan_hj_delta_residual_material_source_6_2_3_05_normalized_windows.json`
shows prescribed L2 `0.319`, `0.127`, and `0.0210` on the early, middle, and
late windows respectively, while the separately fitted normalized rows drift
from roughly `[[0.02, -0.99], [32.2, 12.2]]` early to
`[[8.68, 2.32], [-0.01, -0.07]]` late.  This keeps
`[[6, 2], [3, 0.5]]` as a useful low-dimensional clue, not the finished
source-history law.
A full-field H/J replay was also generated in
`outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_full.h5`
and audited in `outputs\source_history_hj_state_candidate_projection.json`.
Projecting direct internal-state candidates such as `delta*y`, `delta*e`,
`delta*(y-e)`, and their initial-state differences onto the same source moments
does not reproduce the missing trace: a common scalar leaves relative L2 about
`0.998`, while per-trace empirical scales require opposite signs and
`O(10^2-10^3)` raw normalized coefficients.  The remaining law is therefore not
a direct projection of the H/J electric field or Debye memory state.
The high-frequency noIP companion
`examples\hj_highfreq_noip_line_exeyhz_source_centered_axis_aligned.yaml` tests
another tempting shortcut: replacing the low-frequency noIP baseline
(`0.010/0.030 S/m`) with the Debye `sigma_infinity` values
(`0.012/0.036 S/m`).  Its sampled report and source-history summaries
(`outputs\source_history_hj_high_low_noip_sensitivity_comparison.json`) show
that high-minus-low noIP sensitivity is not the missing law either: the best
high-minus-low reference-delta trace still leaves relative L2 `0.678` against
the H/J `delta_residual` trace, with order-1 coefficient signs differing from
the missing source-history coefficients.
The new diagnostic CLI `atem3d.source_history_trace_kernel_cli` reads the
`spatial_time_series` coefficient traces from a matrix-scan report and fits the
zero-initial discrete basis `g_slow - g_fast`, where both `g` terms are BE
relaxations on the saved simulation time grid.  The report
`outputs\source_history_hj_rise_decay_kernel_scan.json` uses the source-centered
H/J `delta_residual` `{0,2}` trace with `slow_tau=1 ms`; the best single
rise-decay candidate has `fast_tau=0.11 ms`, trace-space relative L2 `0.1218`,
per-trace L2 `0.117/0.165`, and normalized coefficients
`[[9.319, 2.677]]`.  This is a better compact time shape than the order-1
`g0/g1` BE trace fit (`0.182`) and the rounded `[[6,2],[3,0.5]]` candidate
(`0.204`), because it starts from zero, rises quickly, then decays.  A
multi-fast-tau fit can drive the residual much lower, but only with normalized
coefficients up to `1.6e9` and column-normalized condition number about
`2.0e12`, so it is a cancellation diagnostic rather than a production law.
The same shape is now encoded as a stricter driven-state skeleton:
`z^{n+1}=a_R z^n + b_R g_slow^{n+1}`, `z^0=0`.  On uniform time steps this is
proportional to `g_slow-g_fast`; in production terms, `z` should be interpreted
as a local MMR/source-recovery state driven by the Debye source memory, not as
an empirical subtraction of two exponentials.
The same CLI now supports `--time-min/--time-max` for window-transfer checks.
The window summary
`outputs\source_history_hj_rise_decay_kernel_window_summary.json` records that
the best single fast time is not constant by window: all-window `0.11 ms`
gives L2 `0.1218`; the early `0.10-0.30 ms` window prefers `0.30 ms` with L2
`0.1418` while the fixed `0.11 ms` choice gives `0.1896`; the middle and late
windows prefer the fastest checked `0.02 ms` with L2 `0.0202` and `0.00647`.
Thus `g_slow-g_fast` is a useful zero-initial convolution shape, but a fixed
empirical `fast_tau` is not the final production law.  The remaining task is
to derive the local H/J MMR/source-recovery operator that produces this
effective build-up and its window/geometry dependence.
A matching sub5 `current_biot` clean-target audit was then generated from the
existing sub5 IP/noIP sampled reports:
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_sub5_delta_residual_hz_line_source_moments_even02_orders1to4.json`.
Its source-trace order-1 BE fit gives L2 `0.1817`, and the rise-decay window
summary
`outputs\source_history_hj_rise_decay_kernel_sub5_window_summary.json` shows
the same all-window build-up scale but different early-window behavior:
all-window best `fast_tau=0.12 ms` with L2 `0.1009`, early-window best
`0.50 ms` with L2 `0.1121`, and middle/late windows again prefer the fastest
checked `0.02 ms`.  This strengthens the interpretation that the `0.1 ms`
scale is a recovery-path clue, while the production law must still be local and
operator-driven rather than a fixed scalar kernel.
A companion `face_basis_cell_biot` sub5 audit uses the same source-centered
H/J geometry and the cleaner `delta_residual` target:
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_face_basis_cell_biot_sub5_delta_residual_hz_line_source_moments_even02_orders1to4.json`.
The main one-shot matrix CLI now reproduces the order-1 case through the
matrix-free static-response path in
`outputs\source_history_matrix_fit_hj_source_centered_axis_aligned_face_basis_cell_biot_sub5_delta_residual_hz_line_source_moments_even02_order1.json`;
it avoids constructing the full dense `face_basis_biot_matrix` and gives the
same receiver-space L2 `0.1967`.  The face-basis static `{0,2}` source-moment
response is rank `2` with projection error `2.327e-4`, so the spatial source
moments are again sufficient.
The face-basis time-kernel evidence is not simpler than `current_biot`: trace
BE orders 1-4 give L2 `0.1949`, `0.1237`, `0.0650`, and `0.0293`, and the
order-1 normalized table is approximately
`[[5.572, 0.881], [5.368, 2.096]]`.  Its rise-decay summary
`outputs\source_history_hj_rise_decay_kernel_face_basis_cell_biot_sub5_window_summary.json`
has all-window best `fast_tau=0.15 ms` with L2 `0.1128`, early-window best
`0.50 ms` with L2 `0.1394`, middle-window best `0.02 ms` with L2 `0.00255`,
and late-window best `0.10 ms` with L2 `0.000366`.  Thus changing the
receiver-side cell reconstruction improves the raw magnetic recovery in some
comparisons, but it does not provide a transferable source-history law; the
missing term is still a constrained H/J MMR/source-recovery operator.
The trace-kernel CLI now also accepts `--basis-kind driven_relaxation`, which
uses the same BE driven state as the runtime `driven_recovery_source_moments`
hook instead of the proportional `g_slow-g_fast` diagnostic column.  The
current-Biot and face-basis sub5 all-window audits are saved in
`outputs\source_history_hj_driven_relaxation_kernel_current_biot_sub5_all.json`
and
`outputs\source_history_hj_driven_relaxation_kernel_face_basis_cell_biot_sub5_all.json`.
As expected on this uniform time grid, the L2 values remain `0.1009` and
`0.1128`, but the normalized one-mode coefficients become
`[[7.727, 1.917]]` and `[[8.221, 1.911]]` in runtime-driven-state units.  This
does not solve the amplitude law, but it removes an avoidable basis-scaling
ambiguity between offline derivation and runtime replay.
The diagnostic runtime hook now accepts this driven-state basis directly through
`kind: driven_recovery_source_moments`.  The example
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_driven_recovery.yaml`
uses the sub1 all-window one-mode coefficients and remains marked
`diagnostic_only: true`.  Its validation report
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_driven_recovery_t010_1ms_validation.json`
reduces the three `Hz` relative L2 values from the uncorrected
`0.588/0.650/0.589` to `0.194/0.205/0.195`, comparable to the fitted order-6
BE diagnostic (`0.199/0.204/0.200`).  The summary is saved in
`outputs\hj_driven_recovery_runtime_validation_summary.json`.  This gives the
future non-fitted FV/MMR derivation a direct runtime validation path.
For larger sub5 diagnostics, the same hook can be applied as a receiver-data
postprocess instead of re-solving the H/J system.  This is valid for the current
diagnostic source-history terms because they only alter magnetic receiver
sampling; they do not feed back into the time-stepping state.  The command

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_history_postprocess_cli `
  outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_data_only.h5 `
  --config examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_driven_recovery.yaml `
  -o outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_sub5_diagnostic_driven_recovery_postprocessed_data_only.h5
```

reproduces the sub5 driven-recovery receiver correction without the PARDISO
memory expansion hit seen in a full re-run.  The subdivision summary in
`outputs\hj_driven_recovery_runtime_subdivision_summary.json` shows sub5 `Hz`
relative L2 improving from `0.414/0.476/0.414` to `0.261/0.291/0.262`, while
`Ex` is unchanged.  This is still diagnostic-only and coefficient-driven; it
does not replace the missing non-fitted local MMR/source-history law.
The source-moment scanner now honors `--source-vector` as well.  Re-running the
same source-centered `delta_residual` audit with `dc_total_current` and
`dc_polarization_current` bases gives the same receiver-space L2 sequence as
the wire basis for orders 1-4 (`0.1937`, `0.1422`, `0.0903`, `0.0532`), saved in
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_dc_total_source_moments_even02_orders1to4.json`
and
`outputs\source_history_matrix_scan_hj_source_centered_axis_aligned_current_biot_delta_residual_hz_line_dc_polarization_source_moments_even02_orders1to4.json`.
So the remaining law is not obtained merely by replacing the wire source moment
with a DC total-current or initial-polarization-current moment.
The fitted upper-bound diagnostic config
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta_residual_order6.yaml`
uses those order-6 coefficients and is marked in validation metadata as
`diagnostic_only: true`.  Its empymod comparison
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta_residual_order6_t010_1ms_validation.json`
reduces `Hz` relative L2 to about `0.199/0.204/0.200`, down from
`0.588/0.650/0.589` for the uncorrected IP run, but it is still a fitted
diagnostic rather than an accepted production law.

SimPEG 0.25.1 does provide H/J TDEM classes
(`Simulation3DMagneticField`, `Simulation3DCurrentDensity`), but they are not
drop-in Debye-IP solvers. The H/J route requires the inverse constitutive
relation with coupled IP memory variables; the derivation and required update
equations are recorded in the H/J formulation section of the implementation
contract. The local inverse Debye coefficients/update are implemented and
tested as `DebyeIPModel.inverse_constitutive_coefficients` and
`DebyeIPModel.inverse_constitutive_update`. The `atem3d.hj` module also
contains an H/J magnetic matrix/RHS/one-step assembly, with the no-IP matrix
and multi-step fields tested against SimPEG's `Simulation3DMagneticField`. The
grounded-wire H/J face source is projected with the same `line_through_faces`
route as SimPEG `LineCurrent.Mfjs`. Layered Debye parameters can be projected to
H/J face dofs with `face_project_debye_model`; `HJMagneticSimulation` advances
multi-step H/J runs, supports `direct`/`cg`/`pardiso` solvers, and samples
`Ex/Ey/Hz` point receivers from the H/J field layout. It supports full-field
output and receiver-only `run_data_only()` validation runs. For cell-centered
layered IP models, H/J assembles the inverse Debye law through cellwise
effective resistivity and face inner products, matching the no-IP H-form
operator ordering at interfaces. H/J magnetic receivers can use the stored
edge `H` field (`magnetic_receiver_mode: stored_h`) or a receiver-side
cell-current Biot recovery (`magnetic_receiver_mode: current_biot`), and the
diagnostic source-history hook supports H/J face-current/face-basis receiver
matrices when coefficients are supplied explicitly or when the
`initial_polarization_source_moments` diagnostic candidate is requested. It also
initializes the long-on-time DC electric field, Debye memories, and MMR magnetic
initial field for grounded sources. Configure this path with `formulation: hj`.
Active CPML is not available in H/J yet, so H/J runs must use large-domain truncation,
`boundary: {kind: none}`, or the outer sponge while CPML remains EB-only.
The `atem3d-time-warp-audit` entry point is diagnostic-only; it reads validation
reports written with `--include-samples` and quantifies whether a residual is
mostly an affine time-axis mismatch:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.time_warp_audit_cli `
  outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_t010_1ms_samples_report.json `
  --components Hz@x=-20 Hz@x=0 Hz@x=20 `
  --scale-min 0.8 --scale-max 1.3 --scale-count 201 `
  --shift-min=-6.0e-5 --shift-max 2.0e-5 --shift-count 161 `
  -o outputs\hj_noip_source_centered_current_biot_sub5_time_warp_audit.json
```

The complementary `atem3d-source-neighborhood-audit` entry point reads a saved
H/J full-field HDF5 result and a sampled validation report, recomputes
`current_biot` from `C h - s_face`, and decomposes magnetic receiver traces by
radial face-current shells around the finite grounded wire:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.source_neighborhood_audit_cli `
  outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_full.h5 `
  outputs\hj_noip_line_exeyhz_source_centered_axis_aligned_current_biot_sub5_t010_1ms_samples_report.json `
  --components Hz@x=-20 Hz@x=0 Hz@x=20 `
  --radial-edges 0 2.5 5 10 20 40 80 `
  --subdivisions 5 `
  --candidates active_source charge_conserving_source dc_initial_current `
  -o outputs\hj_noip_source_centered_current_biot_sub5_source_neighborhood_audit.json
```

## Absorbing Boundary

The main absorbing-boundary route is now an experimental direct time-domain
CPML:

```yaml
boundary:
  kind: cpml
  thickness_cells: 6
  sigma_max: 50.0
  alpha_max: 0.0
  kappa_max: 1.0
  power: 2.0
```

CPML is coupled into the implicit EB time stepper with split-curl memory
variables in both Faraday and Ampere equations.  `thickness_cells: 0` and
`sigma_max: 0` degenerate exactly to the original finite-volume equations in
the test suite.  Active CPML supports the general direct sparse solver and
PARDISO's nonsymmetric mode; CG is reserved for non-CPML runs.  The EB runtime
caches the directional split of `edge_curl` and reuses it in the CPML system
matrix, RHS, and memory updates; without this cache the 50 m/10 m xy6 grid
spends tens of seconds per repeated split on Windows before any physics is
advanced.

The older pragmatic outer sponge remains available through
`boundary: {kind: sponge, ...}` / `make_sponge_sigma`: the outer cells receive a
smooth conductivity increase.  Treat both boundary options as numerical
boundaries that must be validated by convergence sweeps.  For step-off
grounded-source runs, use `apply_to_initial: false` when the sponge is only a
transient absorbing shell; the DC/on-time initial electric and magnetic fields
then use the unsponge physical model.  The sponge can also be limited to named
sides:

```yaml
boundary:
  kind: sponge
  thickness_cells: 4
  strength: 0.1
  apply_to_initial: false
  sides: [x_min, x_max, y_min, y_max, z_min]
```

For the current `z_up` surface-source H/J examples this side/bottom sponge keeps
the physical ground surface out of the artificial conductive shell.

1. increase physical padding,
2. increase CPML/sponge thickness,
3. vary CPML `sigma_max` or sponge strength,
4. confirm receiver curves in the physical region stop changing.

## Run the Example

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.cli `
  examples\small_grounded_wire_debye.yaml `
  -o outputs\small_grounded_wire_debye.h5
```

The output HDF5 contains:

- `times`: time nodes,
- `e`: electric fields (`edges` for EB, `faces` for H/J),
- `b`: EB face magnetic flux density, or `h`: H/J edge magnetic field,
- `data`: receiver data in the order given by the YAML file,
- `config_yaml`: the full input configuration as metadata,
- `formulation`, `electric_field_location`, and `magnetic_field_location`
  attributes documenting the field layout.

For mesh/time/boundary sweeps where only receiver curves are needed, run the
same config without saving field histories:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.cli `
  examples\small_grounded_wire_debye.yaml `
  --data-only `
  -o outputs\small_grounded_wire_debye_data_only.h5
```

The receiver-only HDF5 keeps `times`, `data`, and `config_yaml`, sets
`receiver_data_only = true`, and omits the large `e`, `b`, or `h` datasets.
Use the full-field path when you need later `compare_cli` magnetic-recovery
recomputation from saved `e/b` histories.

For reproducible in-memory empymod sweeps, pass a case-override YAML to
`atem3d.empymod_validation_cli`:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.empymod_validation_cli `
  examples\empymod_validation_noip.yaml `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --sweep-cases examples\validation_sweep_magnetic_recovery.yaml `
  --data-only `
  -o outputs\validation_sweep_magnetic_recovery.json
```

Each case deep-merges its `overrides` into the base config before running. Use
this for magnetic-recovery, time-step, boundary, and refined-core convergence
checks when a full-field HDF5 is not needed.

The saved Debye-IP three-point line used in the magnetic-recovery diagnostics is
also available as a reusable config:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.empymod_validation_cli `
  examples\debye_ip_line_exeyhz_xy6pad.yaml `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --use-config-ip `
  --sweep-cases examples\validation_sweep_debye_ip_magnetic_recovery.yaml `
  --data-only `
  --skip-positive-times 9 `
  -o outputs\debye_ip_line_magnetic_recovery_sweep.json
```

For a quick real-model smoke, use
`examples\validation_sweep_debye_ip_magnetic_recovery_smoke.yaml`; it shortens
the run to twenty `0.01 ms` steps.

The matching active-CPML EB configs for the same strict one-Debye 50 m wire /
10 m offset line are:

- `examples\debye_ip_line_exeyhz_xy6pad_cpml.yaml`
- `examples\debye_ip_line_exeyhz_xy6pad_cpml_stored_b.yaml`
- `examples\debye_ip_line_exeyhz_xy6pad_cpml_biot_stored_b.yaml`

Run and compare the current-Biot CPML case with:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.cli `
  examples\debye_ip_line_exeyhz_xy6pad_cpml.yaml `
  --data-only `
  -o outputs\debye_ip_line_exeyhz_xy6pad_cpml_current_biot_data_only.h5

& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.compare_cli `
  outputs\debye_ip_line_exeyhz_xy6pad_cpml_current_biot_data_only.h5 `
  --depths 0 40 `
  --resistivities 100000000 100 33.33333333333333 `
  --signal -1 --srcpts 51 --recpts 1 --use-config-ip `
  --skip-positive-times 9 --include-samples `
  -o outputs\debye_ip_line_exeyhz_xy6pad_cpml_current_biot_t010_1ms_validation.json
```

The `0.1-1.0 ms` empymod report shows the present production boundary status,
not final magnetic acceptance.  Electric fields are in the same range as the
no-boundary xy6 diagnostic: `Ex` relative L2 is about
`0.081/0.038/0.081` and side `Ey` is about `0.055`; center `Ey` has a huge
relative metric because the reference is near zero.  `current_biot` gives
center `Hz` relative L2 about `0.105`, but side `Hz` remains about `3.51`.
Validation reports now include a `component_groups` summary; for this CPML case
the magnetic group has max relative L2 `3.51`, while the electric group's
relative max is dominated by near-zero center `Ey` even though its absolute Linf
max is `9.56e-4`.
The stored-field CPML checks give side/center `Hz` about `0.617/0.805/0.617`
with Ampere-balanced `b0`, and about `0.553/0.687/0.553` with the diagnostic
Biot-Savart wire `b0`.  Therefore active CPML is operational in EB, but the
full three-point near-source `Hz` acceptance still needs the magnetic
initial/recovery/MMR law; changing the absorbing boundary alone does not close
that gap.

The first refined-core smoke is
`examples\validation_sweep_debye_ip_refined_xy25_smoke.yaml`.  It changes the
x/y core from `5 m` cells to `2.5 m` cells while keeping the same outer padding
and z mesh, again with twenty `0.01 ms` steps.  Use it to verify the refined
sweep path before launching a full `0.1-1 ms` convergence run.

The full 100-step version is
`examples\validation_sweep_debye_ip_refined_xy25.yaml`.  The current report
`outputs\debye_ip_line_refined_xy25_sweep_t010_1ms.json` shows that this x/y
refinement improves the electric components, but strict non-fitted `Hz` changes
only marginally: `current_biot` is about `0.266, 0.402, 0.266` and
`edge_basis_cell_biot` is about `0.249, 0.379, 0.249`.

For larger diagnostic domains where direct sparse factorization is too heavy,
use one of the optional sparse solvers.  `pardiso` is preferred in this Conda
environment when available:

```yaml
solver:
  type: pardiso
```

The fallback iterative option is CG with a Jacobi preconditioner:

```yaml
solver:
  type: cg
  tolerance: 1.0e-7
  maxiter: 10000
  preconditioner: jacobi
```

Repeated time-step blocks can be written as `[dt, repeat]`:

```yaml
time_steps:
  - [0.0001, 20]
  - [0.001, 5]
  - [0.005, 2]
```

To run the SimPEG-parity H/J route instead of the default EB route:

```yaml
formulation: hj
solver:
  type: cg
  tolerance: 1.0e-8
```

A ready smoke config is available at
`examples/hj_grounded_wire_50m_offset10_noip_smoke.yaml`; it exercises the
50 m wire / 10 m offset H/J path and writes `h` on edges instead of EB `b` on
faces.

Two validation-oriented H/J configs use the larger xy6 near-source mesh and
10 microsecond time steps:

- `examples/hj_noip_line_exeyhz_xy6pad.yaml`
- `examples/hj_debye_ip_line_exeyhz_xy6pad.yaml`

Run them through the in-memory validator with `--data-only` to avoid storing
large `h/e` histories.

Initial magnetic fields can be selected explicitly:

```yaml
initial_magnetic_field: ampere
```

Available values are:

- `ampere` (default): discrete vector-potential recovery from the on-time total
  current, enforcing the finite-volume Ampere balance.
- `biot_savart_wire`: open-domain finite-wire vector potential from `geoana`,
  projected to edges and curled so `div B = 0`. This is useful for near-source
  diagnostics because it matches the finite-wire static Biot-Savart scale, but
  it does not include a full grounded-return-current MMR correction and is not
  an accepted `Hz(t)` production mode by itself.
- `zero`: diagnostic zero magnetic start.

For initial-field debugging, write a normal HDF5 result first and then run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.initial_field_diagnostics_cli `
  outputs\empymod_validation_noip_aligned_zhalf_pardiso_dt01ms_biot_wire.h5 `
  --modes ampere biot_savart_wire zero `
  -o outputs\empymod_validation_noip_aligned_zhalf_initial_field_diagnostics.json
```

The report records receiver initial values, `div B`, and the discrete Ampere
residual for each mode.

Magnetic receivers can also use a Biot-Savart current-integral recovery:

```yaml
magnetic_receiver_mode: current_biot
magnetic_recovery_subdivisions: 5
magnetic_recovery_polarization_scale: 1.0
```

Available values are:

- `stored_b` (default): sample `Hx/Hy/Hz` from the stored finite-volume face
  magnetic flux density.
- `current_biot`: recover `H` at receiver points from the finite-volume
  constitutive current. The code forms the edge current
  `M_sigma_inf e - sum_i M_delta_sigma_i y_i`, divides by the unit edge mass,
  averages the resulting edge vector field to cells, and then applies a midpoint
  Biot-Savart volume integral plus the active wire current when the waveform is
  nonzero. This is a receiver recovery/Magnetic-field MMR diagnostic; it does
  not change the internal EB time stepping or the saved `b` field.
- `edge_current_biot`: recover `H` from the same finite-volume edge current
  vector, but evaluate each edge current moment directly at its edge location
  instead of first averaging to cell centers. This is a diagnostic edge-basis
  quadrature for localizing near-source `Hz` error, not a new accepted magnetic
  receiver formula.
- `edge_basis_biot`: recover `H` from the unit-edge-mass current field using
  lowest-order edge-basis reconstruction inside each TensorMesh cell and
  midpoint subcell Biot-Savart integration. This is also diagnostic; on the
  current one-Debye three-point line it does not close the `Hz` mismatch.
- `edge_basis_cell_biot`: reconstruct `E` and Debye memories with edge-basis
  functions inside each cell, then apply that cell's local
  `sigma_inf`/`delta_sigma` before Biot-Savart integration. This avoids
  averaging material/IP parameters onto shared edges first; it is a diagnostic
  step toward a local MMR recovery, not yet the accepted production formula.

`magnetic_recovery_subdivisions` controls midpoint subcell quadrature for the
volume integral.  The default is `1`; near-source diagnostics can use `3` to `5`
for better `Hz` recovery at extra post-processing cost.

`magnetic_recovery_polarization_scale` is a diagnostic knob for Debye-IP
magnetic recovery only.  The default `1.0` is the strict constitutive current
`J = sigma_inf E - sum(delta_sigma y)`.  Changing it does not alter the time
stepping, only recomputed/current-Biot receiver data; use it to isolate whether
an IP `Hz` mismatch is coming from the receiver-side polarization-current
mapping.

`magnetic_recovery_initial_polarization_scale` adds the initial long-on-time
Debye memory to the recomputed magnetic current as
`+ gamma sum(delta_sigma y(0^-))`.  The compare CLI exposes the same diagnostic
through `--magnetic-recovery-initial-polarization-scale`; it is mutually
exclusive with the fitted magnetic-recovery scale options so JSON reports do
not mix prescribed and fitted weights.

For the 50 m wire / 10 m offset diagnostic, late-time `Hz` is sensitive to
physical padding.  The current no-IP reference run improves substantially when
the x/y expanding padding increases from four to six cells per side; treat
padding and CPML sweeps as part of the validation, not optional cleanup.
CPML+PARDISO is verified on the smaller 50 m/IP smoke example, but the refined
52k-edge no-IP target CPML factorization is still too expensive for routine
diagnostics in this environment.
The current 50 m layered CPML magnetic-recovery sweep
`outputs\grounded_wire_50m_offset10_layered_cpml_magnetic_recovery_sweep_skip3.json`
is a failure diagnostic rather than an acceptance report: `Ex` remains about
`0.86-1.05` relative L2 across all magnetic recovery modes, and late-time `Hz`
still has large relative error even when polarization-current scaling reduces
the absolute residual.  CPML/mesh/time-window convergence remains a separate
open gate from the source-history magnetic correction.
Two companion sweeps separate the causes: switching the CPML run from
`ampere` to `biot_savart_wire` initial magnetic field changes almost nothing,
while removing CPML slightly improves `Ex` but worsens late-time `Hz`.  The next
boundary work should therefore be a true padding/mesh/time-window convergence
sweep, not another magnetic-recovery-only tweak.

A time-window-gated in-memory empymod sweep now records this as a machine
check:
`outputs\grounded_wire_50m_offset10_layered_cpml_boundary_timewin_gate.json`.
It uses the 50 m wire / 10 m offset layered-IP smoke mesh, `--use-config-ip`,
`time_min=0.002 s`, `time_max=0.012 s`, relative tolerance `0.25`, and
absolute tolerance `1e-12`.  The sweep fails (`passed=false`).  On this window,
active CPML with `current_biot`/`low_frequency_ratio` gives `Ex` relative L2
about `0.972/0.792/0.972` and `Hz` relative L2 about
`12.23/12.07/12.23` for `x=-20/0/20 m`; removing CPML gives `Ex` about
`0.950/0.774/0.950` and `Hz` about `14.34/14.27/14.34`.  Thus CPML helps the
coarse late magnetic residual but does not fix the electric-field mismatch.

The companion no-boundary time-step sweep
`examples\validation_sweep_time_step_no_boundary_smoke.yaml` writes
`outputs\grounded_wire_50m_offset10_layered_time_step_no_boundary_t012_gate.json`
for the common final gate `t=0.012 s`.  Refining from the original coarse
schedule to uniform `0.5 ms` changes side/center `Ex` relative error from about
`0.996/0.819` to `0.883/0.716`, and side/center `Hz` from about `146/138` to
`138/130`.  Time stepping contributes, but this is not enough for acceptance;
mesh/source/boundary convergence remains the controlling next step.

Two no-boundary mesh-refinement smoke sweeps further narrow the issue:
`examples\validation_sweep_mesh_refinement_no_boundary_smoke.yaml` writes
`outputs\grounded_wire_50m_offset10_layered_mesh_refinement_no_boundary_t012_gate.json`,
and `examples\validation_sweep_mesh_z_refinement_no_boundary_smoke.yaml` writes
`outputs\grounded_wire_50m_offset10_layered_mesh_z_refinement_no_boundary_t012_gate.json`.
At `t=0.012 s` with uniform `1 ms` steps, refining horizontal core cells from
`5 m` to `2.5 m` in x or x/y leaves side/center `Ex` near `0.90/0.73`.
Refining only the vertical core to `2.5 m` gives side/center `Ex` about
`0.896/0.726` and improves `Hz` from about `139/131` to `131/123`, still far
from acceptance.  This points away from simple horizontal core refinement as
the main fix and toward source/receiver convention, primary-secondary
treatment, vertical near-surface/source-depth handling, or larger-domain
boundary convergence.

The diagnostic source-strength sweep
`examples\validation_sweep_source_strength_no_boundary_smoke.yaml` with
`--empymod-strength 1.8` writes
`outputs\grounded_wire_50m_offset10_layered_source_strength1p8_no_boundary_t012_gate.json`.
It makes all three `Ex` receivers pass at `t=0.012 s` (`relative_linf` about
`0.055/0.039/0.055`), confirming that the electric mismatch has a large
source-normalization component.  This is not a production correction:
side `Ey` still fails (`relative_linf` about `0.582`) and `Hz` remains very
large (`O(70)` relative error), so the physical current must remain the config
current while the source/receiver and MMR conventions are derived.
The installed empymod `2.6.0` `bipole` documentation confirms that
`strength != 0` uses the given finite source/receiver length and source
strength, so the physical baseline remains `strength=1 A`.

## empymod Reference Hook

`atem3d.empymod_compare.run_empymod_reference` maps the same finite grounded
wire geometry to `empymod.bipole`:

On this Windows/Conda setup, `empymod` imports more reliably if Numba's cache is
placed in the project directory:

```powershell
$env:NUMBA_CACHE_DIR = (Join-Path (Get-Location) '.numba_cache')
```

```python
from atem3d.empymod_compare import EmpymodSurvey, run_empymod_reference

survey = EmpymodSurvey(
    source_start=(-25.0, 0.0, 0.0),
    source_end=(25.0, 0.0, 0.0),
    receiver_locations=[(0.0, 10.0, 0.0)],
    components=["Ex", "Ey", "Hz"],
    times=times,
    depths=[0.0, 100.0],
    resistivities=[1e8, 100.0, 20.0],
    strength=1.0,
    signal=-1,
)
reference = run_empymod_reference(survey)
```

For a saved ATEM3D result, use:

```powershell
$env:NUMBA_CACHE_DIR = (Join-Path (Get-Location) '.numba_cache')
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.compare_cli `
  outputs\grounded_wire_50m_offset10_layered.h5 `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --receiver-indices 3 `
  --use-config-ip `
  --srcpts 51 `
  -o outputs\validation_report.json `
  --plot outputs\validation_comparison.png
```

`empymod` expects `len(resistivities) = len(depths) + 1`; include the air
half-space as the first resistivity when `depths` contains the ground surface.
The compare command skips the ATEM3D `t=0` initial field by default because
empymod time-domain responses are defined for positive times. A large error in
this smoke workflow is not by itself a solver failure: the ATEM3D model, IP
model, source waveform, boundary treatment, and empymod layered reference must
be made physically identical before interpreting the error values.

For layered empymod validation, set
`model.require_layer_boundary_alignment: true` and choose the mesh origin / `hz`
widths so every empymod depth is also a tensor-mesh z node. Otherwise the finite
volume model assigns conductivities by cell center while empymod uses exact
layer interfaces, so the two problems are not the same.

Notes:

- `Ex` maps to electric receiver azimuth `0`, dip `0`.
- `Ey` maps to electric receiver azimuth `90`, dip `0`.
- `Ez` maps to electric vertical dip; in `coordinate_system: z_up` this is empymod dip `-90`.
- `Hz/Bz/dBzdt` map to magnetic axial vertical orientation; in `coordinate_system: z_up` this is empymod dip `+90`.
- Horizontal magnetic axial components in `coordinate_system: z_up` receive a sign factor of `-1`.
- `Bx/By/Bz` are converted from empymod magnetic-field output by multiplying by `mu0`.
- The compare CLI defaults to `--srcpts 51`; empymod treats finite source coordinates as a center dipole when `srcpts < 3`,
  and the 50 m wire / 10 m offset geometry needs a much denser source quadrature
  for low-frequency finite-source diagnostics.
- Use `--empymod-strength` only as a diagnostic override.  For the finite
  `bipole` comparison with `srcpts >= 3`, the default physical source current
  from the ATEM3D config is the validation baseline; do not multiply the source
  by the 50 m wire length a second time.
- Verify magnetic receiver definitions against a simple halfspace before
  interpreting `Hz` or `dBdt` errors.
- Pass `--use-config-ip` to make empymod use the same fitted Debye conductivity
  dispersion through its `func_eta` hook. Explicit `debye_terms` are copied
  directly; `ip_model: pelton` and `ip_model: cole_cole_conductivity` are first
  converted to the same Debye poles used by the time-domain solver.
- For Pelton resistivity, the fitted Debye model uses the analytic high-frequency
  conductivity `1/(rho0*(1-chargeability))` as the Debye base. Keep this
  convention fixed when comparing to other codes.
- Add `--plot path.png` to write numerical/reference overlays and relative-error
  panels for every receiver component in the validation report.
- Add `--receiver-indices 3` (zero-based data columns) for quick single-column
  smoke checks before running a full receiver-line empymod comparison.  On this
  Windows environment, set `NUMBA_CACHE_DIR` to the local `.numba_cache`
  directory first; otherwise empymod/Numba startup can dominate the run.

## Tests

Run:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m pytest -q
```

The tests cover Debye memory updates, no-IP degeneration, source projection,
sponge generation, CPML split-curl/memory degeneration, solver matrix assembly, YAML loading, HDF5 output, and the
empymod geometry mapping. They also include a no-IP parity test against
`simpeg.electromagnetics.time_domain.Simulation3DElectricField` for the same
grounded line source and time steps.

## Boundary Convergence Benchmark

Run a lightweight smoke benchmark:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.benchmark_cli `
  examples\boundary_benchmark_smoke.yaml `
  -o outputs\boundary_benchmark.json
```

The benchmark spec also supports a compact `base_config` plus per-case
`overrides`, and `data_only: true` for receiver-only convergence sweeps:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.benchmark_cli `
  examples\boundary_benchmark_base_config_smoke.yaml `
  -o outputs\boundary_benchmark_base_config_smoke_report.json
```

Use `absolute_tolerance` together with the relative `tolerance` for components
that are expected to be near zero by symmetry.  A component passes if either its
relative `L_inf` error is below `tolerance` or its absolute `L_inf` error is
below `absolute_tolerance`; the report records the per-component `passed_by`
field.

For production, replace the smoke mesh with a sequence of models that increase
physical padding, sponge thickness, and/or sponge strength. Use the largest or
most conservative case as `reference`; accept the boundary only if the receiver
curves in the physical region change less than the chosen tolerance. Add
`time_min` and/or `time_max` to the benchmark YAML when the acceptance question
is specific to late-time gates:

```yaml
time_min: 0.00091
time_max: 0.001
```

## empymod Validation Scaffold

Start with the no-IP, no-sponge scaffold before validating IP:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.cli `
  examples\empymod_validation_noip.yaml `
  -o outputs\empymod_validation_noip.h5

$env:NUMBA_CACHE_DIR = (Join-Path (Get-Location) '.numba_cache')
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.compare_cli `
  outputs\empymod_validation_noip.h5 `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --srcpts 51 `
  -o outputs\empymod_validation_noip_report.json `
  --plot outputs\empymod_validation_noip_comparison.png
```

This scaffold is intentionally light. For an acceptance run, increase the
physical domain and refine near the 50 m wire and 10 m offset receivers until
the empymod error and boundary benchmark both stabilize.

For receiver-only in-memory sweeps, `atem3d-validate-empymod` can run a config
and write the numerical-vs-empymod report directly:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.empymod_validation_cli `
  examples\empymod_validation_noip.yaml `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --srcpts 51 `
  --data-only `
  --time-min 2.0e-4 `
  --time-max 1.0e-3 `
  --tolerance 0.1 `
  --absolute-tolerance 1.0e-12 `
  --require-pass `
  -o outputs\empymod_validation_noip_inmemory_report.json
```

Use `--time-min` and/or `--time-max` when the acceptance question is tied to a
specific time gate, for example to skip early source-discretization transients
or to isolate late-time boundary effects.  The selected window is recorded in
`metadata.empymod.time_window`.

`--tolerance` is the relative `L_inf` acceptance gate.  Use
`--absolute-tolerance` for near-zero components such as symmetry-line `Ey`,
where a tiny reference amplitude can make relative error uninformative.  The
report records `relative_linf_max`, `absolute_linf_max`, top-level `passed`,
and per-component `passed` / `passed_by` values (`relative`, `absolute`,
`none`, or `not_evaluated` when no gate is requested).
For named sweeps, the top-level `passed` value is the aggregate over all cases:
`false` if any case fails, `null` if at least one case was not evaluated and
none failed, otherwise `true`.  Add `--require-pass` when running the CLI from
a script so failed or unevaluated reports return exit code `1`.

The current no-IP scaffold is a diagnostic workflow. The ATEM3D core has a
unit-level parity test against SimPEG's E-form TDEM for the same no-IP grounded
line source, but coarse finite-domain examples can still disagree strongly with
empymod. Treat large scaffold errors as a signal to check source depth, source
strength, receiver convention (`Hz` versus `dBzdt`), mesh refinement,
physical padding, and boundary convergence.

The validation JSON also stores per-component diagnostics: least-squares scale,
scaled-shape residual, first/last numerical values, first/last references, and
the empymod settings used for the run.

The current z-up smoke runs were regenerated with the configured
`D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D` environment. The no-IP scaffold runs
end-to-end and writes coordinate-system metadata. Earlier coarse one-step
diagnostics suggested `--empymod-strength 50`, but refined time-step checks show
that this double-counts the finite wire length. With the physical `1 A`
strength, source-difference E-form stepping, `biot_savart_wire` initial flux,
`magnetic_receiver_mode: current_biot`, `magnetic_recovery_subdivisions: 5`, and
expanded x/y padding to about `+-1000 m`, the 50 m wire / 10 m offset no-IP
diagnostic gives about `0.248` relative L2 for `Ex@x=0` and `0.0546` for
`Hz@x=0` over `0.1-1 ms`; the last ten gates are about `0.040` for `Ex` and
`0.041` for `Hz`. These are strong diagnostics, not final acceptance results:
CPML/sponge convergence, IP validation, and mesh refinement remain open. The
initial-field diagnostic for the same case records `Hz0=0.00711 A/m` with Ampere
residual `2.5e-13` for `ampere`, and `Hz0=0.01635 A/m` with Ampere residual
`7.67` for `biot_savart_wire`; this is the magnetic-initial-field tradeoff that
motivates the current receiver-side MMR recovery.

The no-IP xy6 run has also been checked on a three-point parallel receiver line
(`x=-20,0,20 m`) with `Ex`, `Ey`, and `Hz`.  `Hz` remains about `5%` relative L2
over `0.1-1 ms` and about `4%` in the last ten gates.  `Ey` is much weaker than
`Ex`; validation reports now include `absolute_linf` so near-zero symmetric
components are not judged only by unstable relative errors.

The H/J magnetic-field route now has a fallback face-source projection for
axis-aligned grounded wires when SimPEG's `line_through_faces` rejects a path
after ambiguous cell-center snapping.  This allows the xy6 no-IP H/J diagnostic
to run with PARDISO on the same 50 m wire geometry.  The high-level H/J source
orientation is now aligned with EB and empymod rather than SimPEG's raw
face-current sign convention.  The resulting 0.1-1 ms empymod report gives
relative L2 about `0.052` for `Ex@x=0` and `0.184` for `Hz@x=0`.  This makes
H/J a live route for the magnetic response, but not yet an accepted replacement
for EB/current-Biot recovery.

The same result is reproducible without full field histories:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.empymod_validation_cli `
  examples\hj_noip_line_exeyhz_xy6pad.yaml `
  --depths 0 40 `
  --resistivities 1e8 100 33.333333333333336 `
  --srcpts 51 `
  --data-only `
  --time-min 0.0001 `
  --time-max 0.001 `
  --tolerance 0.25 `
  --absolute-tolerance 1e-12 `
  -o outputs\hj_noip_line_exeyhz_xy6pad_data_only_t010_1ms.json
```

In that report, all three `Ex` receivers and all three `Hz` receivers pass the
25% relative gate; `Ey` remains a near-zero/symmetry-sensitive component and
does not pass with the strict `1e-12` absolute gate.

On the same xy6 one-Debye IP setup, H/J gives `Ex@x=0` relative L2 about
`0.0125` against empymod, but side `Ex` is about `0.74` and `Hz` remains poor
at about `1.0-1.21` relative L2 even after the cellwise effective-resistivity
assembly.  This confirms that the inverse Debye electric coupling is working in
H/J at the center receiver, while the offset electric and IP magnetic-response
problems are still unresolved:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.empymod_validation_cli `
  examples\hj_debye_ip_line_exeyhz_xy6pad.yaml `
  --depths 0 40 `
  --resistivities 1e8 100 33.333333333333336 `
  --srcpts 51 `
  --data-only `
  --use-config-ip `
  --time-min 0.0001 `
  --time-max 0.001 `
  --tolerance 0.25 `
  --absolute-tolerance 1e-12 `
  -o outputs\hj_debye_ip_line_exeyhz_xy6pad_data_only_t010_1ms.json
```

The offset `Ex` part is now tied to H/J source projection.  On the xy6 mesh, the
old axis-aligned face-source fallback snapped the physical wire at
`(y=0,z=-0.5)` onto one nearby `Fx` channel.  The current fallback linearly
splits current across adjacent transverse face channels and preserves the
wire's transverse first moment; the same current-biot run now gives side `Ex`
relative L2 about `0.278`
(`outputs\hj_debye_ip_line_exeyhz_xy6pad_current_biot_weighted_face_source_t010_1ms_samples_report.json`).
The source-centered H/J mesh in
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot.yaml` removes the
transverse source offset at the mesh-design level.  With
`face_projection: axis_aligned`, the source also preserves symmetric endpoint
support; `outputs\hj_debye_ip_line_exeyhz_source_centered_axis_aligned_current_biot_t010_1ms_samples_report.json`
gives `Ex` relative L2 about `0.088/0.071/0.088` at `x=-20/0/20 m`.
`Hz` still requires the separate MMR/source-history derivation.
The matched stored-field H/J diagnostic
`examples\hj_debye_ip_line_exeyhz_source_centered_stored_h.yaml` was run with
the same empymod settings and saved to
`outputs\hj_debye_ip_line_exeyhz_source_centered_stored_h_t010_1ms_validation.json`.
It leaves the electric errors unchanged but worsens `Hz` to about
`0.781/0.952/0.782`, so the fix is not to bypass the receiver-side MMR recovery
and sample the stored H/J field directly.
The matched no-IP control
`examples\hj_noip_line_exeyhz_source_centered_axis_aligned.yaml` gives `Ex`
about `0.051/0.051/0.051` and `Hz` about `0.161/0.153/0.161`, so the clean
source-centered setup is suitable for IP residual audits.  In that setup the
`Hz` IP residual still projects almost completely onto `{source moment 0,
source moment 2}` at each time (`5.5e-4` spatial projection error), but the
all-window BE time-kernel fit remains poor: orders 1-4 give about
`0.326/0.273/0.197/0.130`, and orders 5-6 improve to `0.0806/0.0493` only with
condition numbers above `1e6`.  The remaining production task is therefore the
early-time source-history/MMR coupling, not another spatial source basis.
Increasing `current_biot` quadrature to sub5 lowers the raw source-centered IP
`Hz` L2 to about `0.414/0.476/0.414`, while `face_basis_cell_biot` sub5 is
slightly worse at about `0.438/0.498/0.439`.  Compared with the earlier
unrefined receiver-recovery paths this is still useful, but the matching sub5
residual scans keep the same spatial answer and poor all-window time fits.
Quadrature and cell-local reconstruction help diagnose the magnetic receiver
operator, but they do not replace the missing source-history/MMR term.  H/J
`current_biot` now caches the explicit
`face_current_biot_matrix`; the source-centered no-IP sub5 data-only run dropped
from about `332 s` to about `133 s` with max data difference `5.6e-17`.

The H/J sponge boundary has a separate source-centered smoke check.  A naive
all-side conductivity sponge is rejected for grounded wires because it either
pollutes the DC/on-time initial field or turns the physical ground surface into
a conductive transient shell.  The current example
`examples\validation_sweep_hj_source_centered_sponge.yaml` uses
`apply_to_initial: false` and `sides: [x_min, x_max, y_min, y_max, z_min]`.
The report
`outputs\validation_sweep_hj_source_centered_sponge_sides_no_top_t010_1ms.json`
keeps `Ex` near the no-boundary result and gives `Hz` relative L2 about
`0.559/0.618/0.560` at `x=-20/0/20 m`, slightly better than the no-boundary
`0.588/0.650/0.589`.  This is still a boundary-convergence candidate, not a
replacement for the missing H/J source-history/MMR term.

The EB `delta6` source-primary diagnostic is also exposed in H/J now, but the
source-centered sweep rejects it as an H/J production formula.  In
`outputs\validation_sweep_hj_source_centered_delta6_t010_1ms.json`, baseline
`current_biot` `Hz` is `0.588/0.650/0.589`; the analytic `wire` delta6 basis
worsens it to `0.862/1.031/0.863`, and the FV `face_current` basis gives
`0.889/1.035/0.890`.  This keeps the diagnostic path testable while ruling out
direct transfer of the EB delta6 normalization to H/J.
The same face-current result is reproduced by the source-history bridge
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta6_source_history.yaml`
after postprocessing the saved receiver-only H/J result; its validation report
`outputs\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_delta6_source_history_t010_1ms_validation.json`
again gives `0.889/1.035/0.890`.  Thus the H/J delta6 failure is not a
postprocess/sign/source-moment issue.

Switching the H/J magnetic receiver improves, but does not close, the IP `Hz`
gap.  With `current_biot` and `magnetic_recovery_subdivisions: 5`,
`outputs\hj_debye_ip_line_hj_current_biot_sub5_t010_1ms.json` gives side
`Hz` relative L2 about `0.62` and center `Hz` about `0.69`.  The more local
`face_basis_cell_biot` mode reconstructs H/J face-`E` and Debye memories inside
each cell before Biot integration; it improves slightly to side `Hz` about
`0.61` and center `Hz` about `0.68`
(`outputs\hj_debye_ip_line_hj_face_basis_cell_biot_sub5_t010_1ms.json`).
Using the non-fitted `low_frequency_ratio` polarization scale in the same H/J
face-basis recovery worsens the center `Hz` to about `1.66`, so it is not the
missing H/J magnetic term for this case.  Adding the initial Debye memory
current with `magnetic_recovery_initial_polarization_scale: 1.0` is much worse
(`Hz` relative L2 about `13-20`), ruling out a direct unscaled initial-memory
correction.

The current best EB/current-Biot Debye-IP baseline is
`outputs\recomputed_current_debye_xy6_low_frequency_ratio_ex_hz_t010_1ms_report.json`.
It uses the saved xy6 fields and `magnetic_recovery_polarization_scale:
low_frequency_ratio`, giving `Ex@x=0` relative L2 about `0.038` and `Hz@x=0`
about `0.105` over `0.1-1 ms`.

The same method has now been run on the requested three-point parallel receiver
line (`x=-20,0,20 m`, `Ex/Ey/Hz`).  The electric components are reasonable
(`Ex` about `0.038-0.081`, `Ey@x=+-20` about `0.055`), but side magnetic
receivers are not accepted: `Hz@x=+-20` is about `3.51` relative L2 even though
`Hz@x=0` remains about `0.105`.  Subdivision scans (`1,3,5`) show this is not
mainly a current-Biot quadrature resolution problem.  Receiver-specific fitted
polarization scales can reduce side and center `Hz` errors to roughly `5-6%`,
but the fitted scales differ (`~0.975` at `x=+-20 m`, `~1.173` at `x=0 m`), so
the missing formula is geometry dependent rather than a single global material
ratio.

The finite-volume edge-basis Biot diagnostic is now exposed as
`magnetic_receiver_mode: edge_basis_biot` and
`compare_cli --recompute-edge-basis-biot`.  With strict
`magnetic_recovery_polarization_scale=1.0`, the three-point one-Debye line gives
the same result as cell-current recovery at one midpoint sample
(`Hz` relative L2 about `0.267, 0.403, 0.267`), while subcell edge-basis
quadrature with `3` to `5` samples worsens the result.  With the inherited
`low_frequency_ratio` polarization scale, the same edge-basis path gives about
`0.734, 0.422, 0.734`.  This rules out simple edge-basis subcell quadrature as
the missing production MMR operator.

The cell-local edge-basis diagnostic is exposed as
`magnetic_receiver_mode: edge_basis_cell_biot` and
`compare_cli --recompute-edge-basis-cell-biot`.  It reconstructs `E` and `y_i`
inside each cell first and only then multiplies by that cell's
`sigma_inf`/`delta_sigma_i`.  On the same one-Debye three-point line, strict
`lambda=1` improves the one-sample `Hz` relative L2 slightly to about
`0.251, 0.381, 0.251`, but subcell quadrature still worsens the curves.  The
old fitted component weights give about `0.052, 0.048, 0.052`, so this is useful
diagnostic evidence for local/component recovery, not a final non-fitted
algorithm.

A stronger diagnostic uses component-wise polarization-current weights in the
magnetic recovery, passed as `--magnetic-recovery-polarization-scale sx,sy,sz`.
For the xy6 one-Debye line, the fitted values
`1.0153363818712349,1.0184707804483801,1.0` give the current best full-line
report:
`outputs\current_debye_line_exeyhz_component_scale_sub1_t010_1ms_report.json`.
Over `0.1-1 ms`, the nonzero electric components are about `0.038-0.081`
relative L2, `Ey@x=+-20` is about `0.055`, and all three `Hz` components are
about `0.015-0.026`.  These weights are still diagnostic because they were fit
from empymod, but they show that a component/geometry-dependent recovery can
close the full-line magnetic mismatch.

The component fit can now be reproduced explicitly instead of hand-tuned:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.compare_cli `
  outputs\current_debye_dt001ms_line_exeyhz_currentbiot_lowfreq_xy6pad.h5 `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --use-config-ip `
  --skip-positive-times 9 `
  --receiver-indices 2 5 8 `
  --recompute-current-biot `
  --fit-magnetic-recovery-component-scale `
  --magnetic-recovery-subdivisions 1 `
  -o outputs\current_debye_line_hz_fit_component_scale_sub1_t010_1ms_report.json
```

The report stores `magnetic_recovery_component_fit` with
`diagnostic_only: true`, the fitted weights, rank, and singular values.  Extra
checks show the fitted target-line weights are not transferable to all receiver
geometries: they work for `x=-20,0,20 m`, `y=10 m`, but degrade `x=+-40 m`,
`y=10 m` where the strict `lambda=1` recovery is already better.  Halving the
time step changes the strict error only modestly.  The reproducible 2.5 m x/y
refined-core sweep now improves the electric components but changes strict
non-fitted `Hz` only marginally, so the current interpretation is
near-source/near-receiver magnetic recovery rather than a finished material-law
correction or a solved grid-convergence path.

The primary-secondary memory diagnostic can also be reproduced from saved
fields. It fits

```text
H = H_ohmic - lambda H_delta_y(t) + gamma H_delta_y(0-)
```

using `--fit-magnetic-recovery-memory-scale`:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.compare_cli `
  outputs\current_debye_dt001ms_biot_wire_eform_currentbiot_sub5_xy6pad.h5 `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --use-config-ip `
  --skip-positive-times 9 `
  --receiver-indices 1 `
  --recompute-current-biot `
  --fit-magnetic-recovery-memory-scale `
  --magnetic-recovery-subdivisions 5 `
  -o outputs\current_debye_hz_x0_fit_memory_scale_sub5_t010_1ms_report.json
```

For the saved one-Debye center receiver this gives
`lambda = 1.2621336257586606`, `gamma = 0.07701959663736895`, and `Hz@x=0`
relative L2 about `0.0309` over `0.1-1 ms`.  A common `lambda/gamma` fit over
all three line `Hz` receivers is not transferable; it improves the side points
but leaves the center at about `0.422` relative L2. Treat this as localization
evidence for the missing MMR/primary-secondary convention, not as an accepted
receiver formula.

The non-fitted Debye backward-Euler value
`gamma = dt / (tau + dt) = 0.009900990099009901` can now be scanned directly
with `--magnetic-recovery-initial-polarization-scale`.  On the saved
three-point one-Debye line with strict `lambda=1`, it reduces side-receiver
`Hz` relative L2 from `0.267` to `0.0519` for `current_biot`, and from `0.251`
to `0.0456` for `edge_basis_cell_biot`.  The center receiver remains poor:
`0.421` for `current_biot` and `0.402` for `edge_basis_cell_biot`.  This is
useful evidence for an initial-memory correction direction, but still not a
transferable production formula.

Two extra root-cause diagnostics are recorded in
`outputs\current_debye_line_hz_projection_source_primary_diagnostics_t010_1ms.json`.
Projecting the edge-current moments to `G^T q = 0` before Biot recovery changes
nothing, so the strict current is already solenoidal in the nodal-gradient
sense.  Fitting a tiny constant finite-wire source-primary `Hz` basis improves
the three `Hz` L2 errors to about `0.048, 0.089, 0.048`, but the fitted scale is
only `-2.78e-5` and has no derived step-off/MMR formula yet; it remains a
localization clue, not an algorithm.

The same report now includes an exponential source-primary diagnostic.  A fit
of `H_wire exp(-t/tau)` with `tau=1 ms` gives `Hz` relative L2 about
`0.036, 0.029, 0.036`; `tau=2 ms` gives about `0.0246, 0.0410, 0.0246`.  The
non-fitted scale `-mu0 sigma_inf L^2` with `2 tau` gives about
`0.029, 0.0376, 0.029`, which is a strong clue for the missing grounded-source
MMR source-primary shape.  It is still not dimensionally closed because
`mu0 sigma L^2` has units of time; this must be derived before it can become
the production correction.

A no-IP consistency check now rules out using that amplitude by itself.  Applying
the same source-primary exponential candidates to the saved no-IP current-Biot
line degrades `Hz` relative L2 from about `0.051, 0.055, 0.051` to ranges of
about `0.115-0.261`, and the optimal-scale residual grows from about
`0.005-0.006` to `0.091-0.231`.  Therefore any production IP correction must
explicitly contain a polarization contrast or memory factor and must vanish in
the no-IP limit.

Simple contrast-bearing amplitudes were also checked.  Replacing the scale by
`-mu0 delta_sigma L^2`, or by the nearby
`-mu0 sigma_inf L^2 (delta_sigma/sigma0)`, moves the strict Debye `Hz` curves in
the right direction but only reduces the three-point L2 values to about
`0.210, 0.325, 0.210` at best for `current_biot` (`0.194, 0.302, 0.194` for
`edge_basis_cell_biot`).  So the missing normalization is not a plain
`delta_sigma` scalar; it has to come from the finite-wire step-off convolution,
the MMR operator, or both.

The exponential initial-memory basis was separated from this source-primary
clue in
`outputs\current_debye_line_hz_initial_memory_exponential_diagnostic.json`.
Using the unit response inferred from strict and `gamma=beta` reports, then
fitting a common `exp(-t/2 tau)` scale, improves the side `Hz` receivers to
about `0.026` for `current_biot` and `0.035` for `edge_basis_cell_biot`, but
the center stays poor at about `0.422` and `0.400`.  Therefore the missing
source-primary/MMR term is not just the initial Debye memory current with a
time kernel.

The static source/MMR normalization was split in
`outputs\current_debye_line_initial_field_diagnostics.json` and
`outputs\current_debye_line_initial_mmr_source_decomposition.json`.  The
Ampere-balanced initial `Hz` is about `0.467` of the low-frequency empymod
finite-wire field at all three receiver positions, while the open-domain
`biot_savart_wire` initial field is about `1.11` of that reference.  The Ampere
field is dominated by the source-vector projection (`~94-95%` of `Hz`), with
conductive return current contributing only `~5-6%`.  This keeps the source
spatial shape alive as a useful basis but rules out return-current splitting as
the centerline Debye `Hz` failure.

The physical IP response was also separated from the numerical recovery
residual in
`outputs\current_debye_line_hz_physical_ip_delta_vs_recovery_residual.json`.
The empymod `IP - noIP` `Hz` difference itself contains a source-primary-like
`H_wire exp(-t/tau)` component.  With `tau=2 ms`, its fitted scale is about
`-7.40e-5`; the ATEM3D numerical `IP - noIP` difference accounts for about
`-4.33e-5`, while the strict residual contributes about `-3.66e-5`.  Their sum
is within about `8%` of the empymod physical scale, so the missing piece is an
under-recovered physical IP source-like magnetic contribution, not an arbitrary
offset.

An empymod-only polarization-strength scan is recorded in
`outputs\empymod_physical_ip_source_primary_scale_vs_polarization.json`.
Keeping `tau=1 ms` and varying `m = delta_sigma/sigma0` from `0.025` to `0.4`,
the physical `IP - noIP` source-like scale is almost linear in `delta_sigma`.
For the `H_wire exp(-t/2 tau)` basis,
`scale / (mu0 delta_sigma L^2)` stays between about `-11.95` and `-11.60`
(mean `-11.80`).  This gives a physically consistent scale law that vanishes
in the no-IP limit: the full physical source-like term is roughly
`-12 mu0 delta_sigma L^2`, and the current strict residual in the `m=0.2`
case is roughly the missing half, `-6 mu0 delta_sigma L^2`.

The half-physical residual candidate is now reproducible through
`compare_cli --magnetic-recovery-source-primary-delta6` and through the runtime
YAML flag `magnetic_recovery_source_primary_delta6: true` for Biot/MMR magnetic
receiver modes.  On the saved three-point Debye line, strict `current_biot`
plus this diagnostic correction gives `Hz` relative L2 about
`0.0291, 0.0376, 0.0291`
(`outputs\current_debye_line_hz_current_biot_delta6_source_primary_sub1_t010_1ms_report.json`).
The same correction on `edge_basis_cell_biot` gives about
`0.0483, 0.0349, 0.0483`
(`outputs\current_debye_line_hz_edge_basis_cell_biot_delta6_source_primary_sub1_t010_1ms_report.json`).
This is the first non-fitted source-primary diagnostic written in terms of
`delta_sigma` rather than `sigma_inf`; it still needs transfer tests before it
can be treated as production.

Those first transfer checks are mixed.  On the saved no-IP center `sub5` report,
`--magnetic-recovery-source-primary-delta6` leaves `Hz@x=0` unchanged at
relative L2 `0.0653`, confirming the intended no-IP degeneration.  On the older
Debye center `sub5` path, it improves `Hz@x=0` from `0.524` to `0.1565`
(`outputs\recomputed_current_debye_dt001ms_biot_wire_eform_currentbiot_sub5_xy6pad_delta6_hz_t010_1ms_report.json`).
On the wider saved `xy8` center `sub5` path, the same correction improves
`Hz@x=0` from `0.518` to `0.1696`
(`outputs\diagnostic_current_debye_dt001ms_biot_wire_eform_currentbiot_sub5_xy8pad_delta6_srcpts51_t010_1ms_report.json`).
Those are substantial non-fitted improvements, but not enough for production
acceptance, so the remaining task is still a true MMR/source-convolution
operator rather than this standalone diagnostic term.

The runtime flag is included in
`examples\validation_sweep_debye_ip_magnetic_recovery.yaml` and its 20-step
smoke companion.  The full `0.1-1 ms` in-memory report
`outputs\debye_ip_line_magnetic_recovery_delta6_sweep_t010_1ms.json`
reproduces the saved-field compare result: `current_biot` `Hz` relative L2 is
`0.0291, 0.0376, 0.0291`, and `edge_basis_cell_biot` gives
`0.0483, 0.0349, 0.0483`.  The smoke report
`outputs\debye_ip_line_magnetic_recovery_delta6_sweep_20step_smoke.json` shows
the in-memory path reduces `current_biot` `Hz` relative L2 from roughly
`0.117, 0.200, 0.117` to `0.022, 0.026, 0.022`, and
`edge_basis_cell_biot` from roughly `0.096, 0.174, 0.096` to
`0.042, 0.022, 0.042`.  This makes mesh/offset/boundary transfer sweeps much
cheaper because full field histories are no longer required for this diagnostic.

The first offset-transfer sweep using that runtime path is
`outputs\debye_ip_delta6_offset_transfer_t010_1ms.json`, with the reusable cases
in `examples\validation_sweep_debye_ip_delta6_offset_transfer.yaml`.  At
`y=20 m`, `current_biot` `Hz` relative L2 changes from about
`0.0959, 0.1456, 0.0959` to `0.0470, 0.0755, 0.0470`; at `y=30 m`, it changes
from about `0.0457, 0.0675, 0.0457` to `0.0285, 0.0441, 0.0285`.
`edge_basis_cell_biot` shows the same direction: `y=20 m` changes from about
`0.0890, 0.1343, 0.0890` to `0.0441, 0.0648, 0.0441`, and `y=30 m` from about
`0.0437, 0.0624, 0.0437` to `0.0282, 0.0392, 0.0282`.  This supports transfer of
the finite-wire `H_wire` spatial shape, but the strict recovery is already much
closer at larger offsets, so the source-primary term remains a diagnostic
near-source correction rather than a production MMR operator.

The runtime source-primary diagnostic also has an FV source-shape option:
`magnetic_recovery_source_primary_delta6_basis: edge_current`.  This replaces
the analytic `LineCurrentWholeSpace` `H_wire` basis with a Biot recovery from
the mesh-projected grounded-wire edge source vector.  The comparison sweep
`outputs\debye_ip_delta6_source_basis_t010_1ms.json` shows it is very close to
the analytic basis on the current grid: for `current_biot`, `Hz` relative L2 is
`0.0291, 0.0376, 0.0291` with the analytic `wire` basis and
`0.0303, 0.0376, 0.0303` with `edge_current`; for `edge_basis_cell_biot`, it is
`0.0483, 0.0349, 0.0483` versus `0.0497, 0.0350, 0.0497`.  This is a useful
bridge toward a finite-volume-compatible MMR/source operator, but it is still a
diagnostic source shape, not the final derived convolution law.

The raw stored-field Ampere residual was also tested as a possible derived MMR
source:
`outputs\current_debye_line_hz_ampere_residual_biot_diagnostic.json`.
The residual current was computed as `C^T M_mu^-1 b - j_c - s_e` and recovered
with the edge-current Biot kernel.  As a unit correction it is thousands of
times too large (`Hz` relative L2 blows up to about `1969, 3491, 1969`).  A
least-squares fitted scale of only about `-1.1e-4` to `-1.3e-4` improves the
strict current-Biot result to about `0.047, 0.087, 0.047`, still weaker than the
non-fitted `delta6` source-primary term.  This rules out simply adding the raw
Ampere residual as the production MMR correction; the missing term must come
from a source/history convolution, not from the stored-field residual alone.

That negative result now has a sharper FV-source projection check.  The CLI
`atem3d-ampere-source-projection` reads saved EB `e/b` HDF5 results, computes
`C^T M_mu^-1 b - j_c - s_e`, and projects it onto the initial grounded-source
edge vector without using empymod.  It also accepts H/J `e/h` HDF5 results and
uses the signed face-source convention `s_face=-source.initial_face_vector`,
checking `C h - q(E,y) - s_face` against the face source vector.  The H/J
initial MMR test now verifies `C h0 = j0 + s_face` and
`D(j0+s_face)=0` to machine precision, while the opposite source sign leaves an
`O(1)` residual.  The saved H/J full-field pair
`outputs\hj_debye_xy6_dt001ms_100step_pardiso_signfix.h5` minus
`outputs\hj_noip_xy6_dt001ms_100step_pardiso_signfix.h5` now has the matching
report `outputs\hj_ampere_source_projection_debye_minus_noip_be_basis.json`;
its normalized source-projection coefficients are only about `1e-13`, with BE
`g0/g1` coefficients about `-1.3e-13, -1.0e-13`.  Therefore the H/J missing
`Hz` source-history term is not hiding in the discrete Ampere residual.  The
source-history matrix CLI can now read such a candidate JSON directly through
`--prescribed-coefficients-file`, select a dotted key such as
`discrete_basis_fit.coefficients`, and place that one-dimensional history series
into a chosen source-moment trace with `--prescribed-spatial-trace-index`.
The report
`outputs\source_history_matrix_hj_delta_residual_ampere_projection_candidate.json`
uses this bridge to put the H/J Ampere-projection `g0/g1` coefficients into the
degree-zero source-face moment for the source-centered `Hz delta_residual`
audit.  Both receiver-space and trace-space prescribed relative L2 are `1.0`,
while the optimal order-1 trace fit is about `0.193`, so the candidate is
rejected in the same prescribed path reserved for future derived FV/MMR laws.
The lower-level projection step is now public as
`source_history_operator.project_vector_to_spatial_basis`: it projects any local
FV candidate vector onto source/spatial moments either in DOF space or after the
selected MMR receiver matrix.  The runtime initial-polarization diagnostic now
uses this shared operator, so future local FV/MMR candidate vectors can be
converted to prescribed source-history coefficients without another private
least-squares path.  The matrix CLI also exposes this first built-in candidate
through `--prescribed-candidate initial_polarization`.  On the source-centered
H/J `Hz delta_residual` audit,
`outputs\source_history_matrix_hj_delta_residual_initial_polarization_candidate.json`
uses receiver-space projection, records `requires_ip: true`, and gives
prescribed relative L2 `63.2` (`364` in trace space), with normalized `g0`
coefficients about
`942, -2811`; the matching
`outputs\source_history_matrix_hj_delta_residual_initial_polarization_candidate_dof_l2.json`
uses DOF projection and is also rejected (`81.1` receiver-space L2 and `281` in
trace space).  This keeps
initial polarization as a documented negative candidate, not a production
source-history law.  A stricter non-fitted variant now first projects
`-delta_sigma*y0` through the H/J face-current continuity equation
`D(j_p - M_rho_inf^-1 G phi)=0` before source-moment projection.  It is exposed
as `charge_conserving_initial_polarization_source_moments` and demonstrated in
`examples\hj_debye_ip_line_exeyhz_source_centered_current_biot_diagnostic_charge_conserving_initial_polarization.yaml`.
The same candidate is now available in the matrix CLI as
`--prescribed-candidate charge_conserving_initial_polarization`; its
`delta_residual` reports record `requires_ip: true` and are rejected with
receiver-space prescribed L2 `66.9` (`382` trace) for receiver projection and
`33.7` (`77.2` trace) for DOF projection.
Postprocessed validation rejects it as well: `Hz` relative L2 is about
`25.46/28.46/25.46`, and the opposite-sign control is still about
`24.35/29.68/24.35`.  Thus enforcing charge conservation on the initial
polarization current alone does not produce the missing H/J MMR/source-history
operator.  A related source-path memory check treats the impressed face source
as an ordinary low-frequency material electric field,
`E_source0 = M_rho0 s0 / M_unit`, and projects `delta_sigma*E_source0` onto the
same moments.  The audit
`outputs\source_history_source_path_memory_candidate_trace_audit.json` rejects
it by scale: the normalized main coefficient is about `-3.18e4`, while the
target trace is `O(1-4)`, and tested decay/rise-decay histories leave L2 above
`3.2e3`.  An offline high-minus-low DC grounded-current difference check gives
only `1e-9`-scale source-moment coefficients and leaves `Hz` at the uncorrected
`0.588/0.650/0.589` level, so the missing term is not a simple
`sigma_0`-versus-`sigma_infinity` DC source-current difference either.  On
`outputs\current_debye_dt001ms_line_exeyhz_currentbiot_lowfreq_xy6pad.h5`, the
report `outputs\ampere_source_projection_current_debye_line_be_basis.json`
finds normalized source-projection coefficients near `-5.0e5`; its BE
`g0/g1` fit gives about `-4.60e5, -7.98e5`.  This is orders of magnitude away
from the empirical source-history coefficients near `-6, -2`, so even the
source-vector projection of the raw Ampere residual is not the missing MMR
operator.
Using `--subtract-baseline` with the matching no-IP field history cancels that
large common source-convention term:
`outputs\ampere_source_projection_debye_minus_noip_be_basis.json` leaves only
`1e-10` to `2.5e-9` in normalized projection, with BE coefficients about
`7.5e-11, 5.6e-9`.  Thus the missing IP source-history term is also not the
Debye-minus-noIP Ampere projection.

The source-primary exponential fit is now a reproducible diagnostic instead of
an ad hoc script.  `atem3d-source-primary-fit` reads two validation reports with
stored `samples`, forms targets such as `reference_delta = IP - noIP`, computes
the finite-wire source `H` from the saved config, scans
`H_source exp(-t/tau_kernel)`, and records both the fitted scale and
`scale/(mu0 delta_sigma L^2)`.  The report
`outputs\source_primary_fit_reference_delta_from_samples.json` reproduces the
physical reference IP-noIP fit on the three-point line: for
`tau_kernel=2 ms`, `scale/(mu0 delta_sigma L^2) = -11.78` with aggregate
relative L2 `0.227`; extending the same scan to `4 ms` lowers this particular
single-tau basis fit to L2 `0.197` and normalized scale `-10.47`.  This is a
better audit trail for the source-kernel clue, but it is still evidence for the
missing convolution operator, not the operator itself.

A direct empymod-only tau-transfer scan is also available as
`atem3d-empymod-source-primary-scan`.  It computes the no-IP and Debye-IP
empymod references first, then imports `geoana` only after those calls to avoid
a Windows/Conda hang observed when `LineCurrentWholeSpace` is imported before
finite-source empymod time-domain calls with larger `srcpts`.  The report
`outputs\empymod_source_primary_tau_scan_t010_1ms_srcpts51_cli.json` scans
Debye `tau = 0.5, 1, 2, 5 ms` over `0.1-1 ms` for the three `Hz` receivers.
It shows that a universal `tau_kernel = 2 tau` is not supported by this
evidence: the best checked kernels are `0.75 ms`, `4 ms`, `8 ms`, and `20 ms`,
with relative L2 `0.106`, `0.197`, `0.510`, and `0.994`.  The source-primary
shape remains useful near `tau=1 ms`, but the true production correction must
come from a derived source/history operator, not a fixed tau multiplier.
The same report also stores a non-parametric empirical source kernel `K(t)` from
`target(t,r) ~= K(t) H_source(r)`.  The spatial source shape alone explains the
three-receiver physical IP-noIP increment well for `tau=0.5 ms` and `1 ms`
(`K(t)` shape residuals `0.025` and `0.039`), but the normalized kernel is not a
single monotone exponential; for `tau=2 ms` and `5 ms` the early-time kernel
even changes sign over the `0.1-1 ms` window.  This points to a real
source-history convolution rather than a reusable scalar decay.

The same empirical-kernel extraction is now available in
`atem3d-source-primary-fit` for sampled validation reports.  On the strict
`current_biot` one-Debye `Hz` residual
(`outputs\source_primary_fit_strict_current_biot_ip_residual_from_samples.json`),
the residual is strongly source-shaped: the empirical `K(t) H_source` fit has
relative L2 `0.056`.  If that empirical residual is added back to the numerical
data, the three `Hz` relative L2 errors become about `0.0144, 0.0236, 0.0144`.
This is not a production correction because `K(t)` was extracted from empymod,
but it sharply localizes the missing formula to the grounded-source history
kernel.

The empirical `K(t)` can now be reduced further with a diagnostic basis fit
using `--kernel-basis-tau`, `--kernel-basis-powers`, and optionally
`--kernel-basis-include-constant`.  The fitted form is a constant plus
`(t/tau)^p exp(-t/tau)` terms.  For the same strict `current_biot` residual,
`outputs\source_primary_fit_strict_current_biot_ip_residual_basis_from_samples.json`
fits the normalized empirical kernel with `p=0,1` and `tau=1 ms`, giving
coefficients `-6.364` and `-2.203` in units of `mu0 delta_sigma L^2`, with
relative L2 `0.0563`.  This supports an exponential-polynomial source-history
structure, but the coefficients are still empirical and must be derived from
the discrete Debye/source update before becoming runtime physics.

The same CLI can also fit `K(t)` to discrete backward-Euler Debye history bases
with `--kernel-basis-discrete --kernel-basis-max-order 1`.  These bases are
computed from the actual config `time_steps`: order zero is the homogeneous
Debye relaxation, and order one is the same-pole cascade corresponding to the
continuous-limit `(t/tau) exp(-t/tau)` term.  On the strict residual report
`outputs\source_primary_fit_strict_current_biot_ip_residual_be_basis_from_samples.json`,
the normalized coefficients are `-6.366` and `-2.182`, with relative L2
`0.05631`.  This makes the next derivation target sharper: the missing MMR
source-history term should be expressible through discrete Debye relaxation
and cascade states driven by the grounded-source step-off, not through an
externally fitted continuous kernel.

Recovery-path transfer is mixed, which is important for the derivation.  On the
`edge_basis_cell_biot` three-point residual, the BE-basis report
`outputs\source_primary_fit_edge_basis_cell_biot_ip_residual_be_basis_from_samples.json`
gives normalized coefficients `-5.409, -3.343` and residual L2 `0.0914`.
On the older `sub5` current-Biot path,
`outputs\source_primary_fit_current_biot_sub5_ip_residual_be_basis_from_samples.json`
gives `-11.389, 3.092` and residual L2 `0.0429`.  Thus the discrete Debye
history basis is a useful coordinate system, but its coefficients are not
universal constants across recovery paths.  A production correction must derive
the MMR/source coupling coefficient from the same discrete current recovery
used at the receiver.

The source-basis part is less problematic.  `atem3d-source-primary-fit` now
accepts `--source-basis edge_current`, which replaces the analytic finite-wire
`H_wire` basis with the mesh-projected grounded-source edge vector recovered by
the edge-current Biot operator.  On the strict `current_biot` residual,
`outputs\source_primary_fit_strict_current_biot_ip_residual_edge_current_be_basis_from_samples.json`
gives normalized BE coefficients `-6.340, -2.176`, very close to the analytic
wire basis `-6.366, -2.182`.  This points the remaining discrepancy at the
MMR/recovery-path coupling rather than at the static source spatial basis.

The source-primary shape was cross-checked on the older center-only `sub5`
report.  A fitted `H_wire exp(-t/tau)` correction still works (`Hz@x=0` L2
about `0.0419` for `tau=1 ms`), but the non-fitted
`-mu0 sigma_inf L^2` amplitude does not transfer (`0.156-0.211`).  Therefore
the spatial/time shape is useful, while the normalization is still unresolved.
On the same three-point line with `edge_basis_cell_biot` and one midpoint
sample, the non-fitted `-mu0 sigma_inf L^2` amplitude with `2 tau` gives
`0.048, 0.0349, 0.048`, so the source-primary exponential shape is not specific
to the original cell-current averaging path.
Per-receiver fits show the `sub1` side and center amplitudes differ only
modestly, but the older `sub5` center-only path needs a much larger amplitude;
the missing normalization is therefore recovery-path sensitive.

To expose that geometry dependence directly, use
`--fit-magnetic-recovery-memory-scale-per-receiver`.  On the saved three-point
line it gives independent `Hz` relative L2 values of about
`0.0217, 0.0140, 0.0217` for `x=-20,0,20 m`.  The side receivers fit
`lambda ~= 0.9874`, `gamma ~= 0.00287`, while the center receiver fits
`lambda ~= 1.2287`, `gamma ~= 0.00904`.  This is strong evidence that the
remaining correction is geometric/local, not a transferable bulk IP material
law.

Two follow-up checks sharpen that interpretation. Excluding the receiver's
containing cell or local cell neighborhood from the Biot volume integral does
not improve the one-Debye target `Hz`, so a simple self-cell principal-value
patch is not the answer. A `1.25 m` reduced-core mesh improves the side
receivers but not the centerline, which points to a real refined-core/MMR
convergence study rather than another scalar or component fit.  The empymod
helper can now build surveys directly from a config and supplied time array,
which is the intended path for larger in-memory mesh/time/boundary sweeps.

A first one-Debye IP run on the same six-padding geometry uses
`sigma_inf - delta_sigma` equal to the no-IP conductivities and compares to
empymod through `--use-config-ip`.  In the `0.1-1 ms` window it gives
`Ex@x=0` relative L2 about `0.038`, while `Hz@x=0` remains about `0.50`; this
supports the Debye electric-field coupling but leaves IP magnetic recovery as an
open validation item. Expanding x/y padding further to about `+-1634 m` does not
improve the IP `Hz` mismatch, so this is not currently behaving like the no-IP
finite-boundary error.

The current code now reconstructs `current_biot` currents from the
finite-volume edge current using the unit edge-mass projection. Receiver-only
recompute reports were generated with the new compare path:
`outputs/recomputed_current_noip_dt001ms_biot_wire_eform_currentbiot_sub5_xy6pad_srcpts51_t010_1ms_report.json`
gives `Hz@x=0` relative L2 about `0.065`, while
`outputs/recomputed_current_debye_dt001ms_biot_wire_eform_currentbiot_sub5_xy6pad_srcpts51_t010_1ms_report.json`
keeps `Ex@x=0` at about `0.038` but gives `Hz@x=0` about `0.524`. The
remaining IP magnetic problem is therefore deeper than this projection detail.
The new diagnostic `magnetic_recovery_polarization_scale` narrows that problem:
on the same saved Debye-IP field file, recomputing only `Hz@x=0` with
`current_biot`, `magnetic_recovery_subdivisions: 5`, and scale
`1.1726210524361325` gives relative L2 about `0.065` over `0.1-1 ms`.  This
matches the no-IP error scale, but it is not yet a production formula because
the scale was obtained from an empymod fit; the next step is to derive this
primary-secondary/polarization-current weighting rather than tune it.
The companion diagnostic
`outputs/magnetic_recovery_mode_diagnostics_xy6pad.json` also rules out the
tested stored-`b`, stored-`b - b0`, one-step `b` delta, Ampere-balanced
vector-potential reconstruction, and simple Debye polarization-current scaling
variants for this window.
A z-padding sweep is recorded in
`outputs/debye_ip_z_padding_sensitivity_xy6_currentbiot.json`: increasing the
Debye-IP mesh from `z=[-680,240] m` to `[-1320,480] m` improves `Hz@x=0` from
about `0.524` to `0.486`, but pushing to `[-2600,960] m` only reaches about
`0.482`. Vertical truncation matters, but it is not the main missing IP magnetic
piece.

`dBdt` remains unresolved because ATEM3D currently samples a backward-Euler
step-average derivative, including the `t=0^-` static magnetic field on the
first step, while empymod's `mrec='b'` gives a positive-time switch-off
`dB/dt` response.  Do not treat these reports as proof of analytic correctness;
they are diagnostic artifacts for the next magnetic initial-field,
receiver-convention, solver, and boundary work.

For `dBdt` diagnostics, `compare_cli` also supports `--skip-positive-times N`.
This is useful for excluding the first post-step-off backward-Euler derivative,
which contains the transition from the on-current static magnetic field.

For magnetic-recovery experiments, `compare_cli` can recompute receiver data
from the saved `e`/`b` fields before comparing to empymod:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.compare_cli `
  outputs\current_debye_dt001ms_biot_wire_eform_currentbiot_sub5_xy6pad.h5 `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --receiver-indices 1 `
  --use-config-ip `
  --skip-positive-times 9 `
  --recompute-current-biot `
  --magnetic-recovery-subdivisions 5 `
  --magnetic-recovery-polarization-scale low_frequency_ratio `
  -o outputs\recomputed_current_biot_report.json
```

This is a receiver-side post-processing comparison; it does not rerun the
implicit time stepping. Use it to test MMR/current-recovery changes against
saved large simulations before spending time on another full solve.
Use `--recompute-edge-current-biot` instead of `--recompute-current-biot` to
test the edge-current-moment quadrature on the same saved fields.  On the saved
xy6 one-Debye three-point line, this edge quadrature gives `Hz` relative L2
about `2.07, 0.253, 2.07` for `x=-20,0,20 m` over `0.1-1 ms`, so it currently
rules out simple edge-point quadrature as the missing production MMR formula.
The symbolic `low_frequency_ratio` scale applies the local
`sigma_inf / (sigma_inf - sum(delta_sigma))` weighting in the recovered
polarization current. On the saved Debye-IP xy6 diagnostic, the generated
`outputs\recomputed_current_biot_low_frequency_ratio_smoke_report.json` gives
`Hz@x=0` relative L2 about `0.105` over the 0.1-1 ms window.

For current full-field runs, `atem3d.magnetic_recovery_decomposition_cli` can
rerun a YAML config and split every magnetic receiver's Biot recovery into
`ohmic_current`, `polarization_memory`, `initial_polarization`,
`source_current`, `source_primary_delta6`, and `source_history` contributions:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.magnetic_recovery_decomposition_cli `
  examples\debye_ip_line_exeyhz_xy6pad_cpml.yaml `
  --time-min 1.0e-4 `
  --time-max 1.0e-3 `
  -o outputs\magnetic_recovery_decomposition_report.json
```

The report is marked `diagnostic_only`.  Its purpose is to verify that the
runtime magnetic data equal the sum of the currently implemented recovery
channels and to identify which channel dominates the empymod residual.  It is
not a new MMR/source-history law.  In no-IP or zero-`delta_sigma` cases the
aggregate IP terms remain zero by construction, so any future production
Debye-IP source-history correction must still prove a separate
`delta_sigma`-dependent formula.
If the validation JSON was written with `--include-samples`, pass it back with
`--validation-report` to project the required missing response
`reference - numerical` onto each recovery channel:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.magnetic_recovery_decomposition_cli `
  examples\debye_ip_line_exeyhz_xy6pad_cpml.yaml `
  --time-indices 10 `
  --validation-report outputs\debye_ip_line_exeyhz_xy6pad_cpml_current_biot_t010_1ms_validation.json `
  -o outputs\debye_ip_line_exeyhz_xy6pad_cpml_current_biot_t010_decomposition_alignment.json
```

For that single `0.1 ms` CPML gate, the runtime decomposition recombines to the
validation numerical data with relative L2 `1.5e-13`.  The best single residual
direction is `polarization_memory`, with scalar `-0.186`, cosine `-0.992`, and
post-projection relative L2 `0.124`.  A two-channel least-squares combination of
`ohmic_current` and `polarization_memory` fits the three `Hz` receiver residuals
to roundoff at that one gate.  This is useful evidence for the missing
receiver/source-channel coupling, but because it uses an empymod residual and
only one time gate it remains diagnostic-only.
The same CLI now also writes `per_time_alignment`, so the scalar/cosine trend
can be inspected before proposing a time kernel.  A four-gate CPML report is
saved in
`outputs\debye_ip_line_exeyhz_xy6pad_cpml_current_biot_t010_1ms_decomposition_alignment_4gates.json`.
Over `0.1, 0.2, 0.5, 1.0 ms`, the global best single direction remains
`polarization_memory` with scalar `-0.1857` and relative L2 `0.1188`.  The
per-gate `polarization_memory` scalars are approximately
`-0.1861, -0.1876, -0.1835, -0.1815`, with cosine magnitude above `0.992` at all
four gates.  At the final gate `ohmic_current` is a slightly better single
direction (`0.0886` versus `0.0944` relative L2), so the diagnostic points to a
stable polarization-memory weighting plus late-time recovery/boundary mixing,
not yet to a complete production MMR/source-history law.
The residual scalar can also be read as a multiplier on the currently included
polarization-memory term: `1 + scalar`.  For the four-gate report this multiplier
is `0.8143`; because the config used `magnetic_recovery_polarization_scale:
low_frequency_ratio`, and the two conductive layers both have
`sigma_inf / sigma_0 = 1.2`, the inferred physical scale is
`0.8143 * 1.2 = 0.9772`.  The per-gate inferred scales are about
`0.9766, 0.9749, 0.9798, 0.9823`.  This is close to the strict unscaled
polarization current (`scale = 1.0`) and is now a concrete derivation target:
prove whether the receiver-side Debye memory should use the unscaled
`-delta_sigma y` current, and then revalidate across the full window/components.
Using the same four-gate decomposition without rerunning the solver, replacing
the configured `low_frequency_ratio` memory contribution by strict `scale = 1`
corresponds to adding `(1/1.2 - 1) * polarization_memory = -0.1667 *
polarization_memory`.  Over the twelve `Hz` samples in the four-gate report,
the raw low-frequency-ratio data have relative L2 `1.43` to empymod, the strict
`scale = 1` replay reduces that to `0.223`, and the best fitted
polarization-memory scalar gives `0.170`.  Thus strict scaling captures most of
this CPML diagnostic residual, but not all of it; it is a strong formula
candidate to test across the full `0.1-1 ms` window, not an acceptance result.
That full-window rerun is now saved as
`examples\debye_ip_line_exeyhz_xy6pad_cpml_strict_polscale.yaml`, which changes
only `magnetic_recovery_polarization_scale` from `low_frequency_ratio` to
strict `1.0`.  The data-only solve is
`outputs\debye_ip_line_exeyhz_xy6pad_cpml_strict_polscale_data_only.h5`, and
the same `compare_cli --include-samples` validation over the 91 gates from
`0.1-1.0 ms` is
`outputs\debye_ip_line_exeyhz_xy6pad_cpml_strict_polscale_t010_1ms_compare.json`.
The combined three-receiver `Hz` relative L2 drops from `2.926` for
`low_frequency_ratio` to `0.437` for strict `scale = 1.0`; the side receivers
drop from `3.510/3.510` to `0.392/0.392`.  This is also better than the
stored-`B` CPML baselines (`0.680` for Ampere initial field and `0.597` for
Biot-Savart-wire initial field, combined `Hz`).  The center receiver is the
important caveat: `Hz@x=0` changes from `0.105` to `0.525`.  Therefore the
full-window run supports using the strict constitutive Debye polarization
current `J_p = -delta_sigma y` in receiver-side magnetic recovery, but it also
shows that the near-source center-channel/source-history/MMR term is still
unresolved and must not be replaced by a fitted scalar correction.

For mesh/time/boundary sweeps where full field histories are not needed, use
the in-memory empymod validator. It runs the config, compares `result.data`
directly to empymod, and writes the same component error structure without
first creating an HDF5 result:

```powershell
& 'D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe' -m atem3d.empymod_validation_cli `
  examples\grounded_wire_50m_offset10_layered.yaml `
  --depths 0 40 `
  --resistivities 100000000 100 33.3333333333 `
  --use-config-ip `
  --skip-positive-times 1 `
  --data-only `
  -o outputs\grounded_wire_in_memory_empymod_report.json
```

This path is intended for convergence sweeps. Use the HDF5 `compare_cli` path
when you need saved `e/b` fields for receiver-side recomputation diagnostics.

## Current Scope

Implemented:

- grounded finite wire source,
- direct time-domain finite-volume EB stepping,
- DC/on-time initial electric field and selectable magnetic initial fields for
  step-off source,
- single or multiple Debye IP terms,
- layered media through z-interval YAML definitions,
- optional conductivity Cole-Cole or Pelton resistivity fitting to Debye poles,
- parallel receiver-line expansion for the 50 m wire / 10 m offset survey layout,
- edge/face point receivers for `Ex`, `Ey`, `Ez`, `Bx`, `By`, `Bz`, `Hx`, `Hy`, `Hz`,
- outer sponge conductivity,
- experimental direct time-domain CPML split-curl coupling,
- H/J direct time-domain magnetic formulation with inverse Debye coupling and
  receiver-only validation runs,
- YAML-driven runs,
- full-field and receiver-only HDF5 output,
- empymod mapping and error-summary helpers,
- receiver-only in-memory empymod validation for larger mesh/time sweeps.

Still required for production-grade interpretation:

- quantitative convergence studies against empymod,
- robust Cole-Cole/Pelton-to-multi-Debye fitting,
- quantitative CPML and sponge boundary-convergence studies,
- grid refinement studies for the 50 m wire and 10 m offset near-source survey,
- validation of source waveform, receiver sign, and unit conventions.
