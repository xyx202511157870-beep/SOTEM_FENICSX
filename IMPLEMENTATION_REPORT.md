# P0-P8 Implementation Report

## Scope

This report covers the implementation rounds currently committed or staged from the task book:

- P0: freeze the current total-field baseline.
- P1: add waveform interfaces, full turn-off time grid, and interval-average `dI/dt`.
- P2: add no-IP three-component validation artifacts and diagnostics.
- P3 partial: add a Debye/Prony material conductivity API and pure-Python memory tests.
- P4 partial: add primary-field provider interfaces with zero and cached providers.
- P5 partial: add a DC secondary initialization core with zero-contrast tests.
- P6 partial: add no-IP/IP primary-secondary TDEM step kernels with zero-contrast tests.
- P7 partial: add no-IP/IP three-component validation smoke artifact writer.
- P8 partial: add complex-terrain leakage-channel material-map smoke utilities.
- P7 CLI: add `validate-noip-3comp` and `validate-ip-3comp` artifact commands from CSV inputs.

It does not claim that the full 1e-5 s to 1 s 5% accuracy target is achieved.

## Implemented Modules

- `src/atem3d/waveforms.py`
  - `Waveform`
  - `StepOffWaveform`
  - `LinearRampOffWaveform`
  - `TabulatedWaveform`
  - `build_internal_time_grid`

- `src/atem3d/metrics.py`
  - `robust_relative_error`
  - `robust_component_errors`

- `dolfinx/sotem_pipeline.py`
  - Added `min_steps_during_turnoff`.
  - Added `min_steps_before_first_observation` for after-ramp internal
    substeps between ramp-off end and the first observation output.
  - Replaced point-value source derivative with interval-average `dI/dt`.
  - Updated after-ramp internal schedule to include turn-off history before observation times.
  - Added receiver sampling modes:
    - `--receiver-type point`
    - `--receiver-type volume_average`
    - `--receiver-type disk_average`
    - `--receiver-average-radius R`
    - `--receiver-diagnostic-types point,disk_average`
    Average receiver modes evaluate deterministic diagnostic sample points and
    average the per-point collapsed cell candidates.
    `--receiver-diagnostic-types` writes simultaneous point/average diagnostic
    receiver rows to `receiver_diagnostics.csv` and automatically plots
    `receiver_diagnostics.png` without changing `predictions.csv`.
    When empymod/reference data are available, diagnostic receiver responses
    are also compared against the reference and written to
    `receiver_reference_errors.csv` and `receiver_reference_error_curves.png`.
  - Added P2 validation artifacts:
    - `predictions.csv`
    - `reference_empymod_or_1d.csv`
    - `errors.csv`
    - `error_summary.json`
    - `comparison_3comp.png`
    - `error_curves_3comp.png`
    - `diagnostics.json`
    - `run_config_resolved.yaml`
  - Added automatic diagnostic scaffolding and source waveform consistency diagnostics.
  - Receiver diagnostics now record sampling/cell-candidate statistics:
    `sample_count`, `candidate_count_min`, `candidate_count_max`, and
    `candidate_count_mean` in `receiver_diagnostics.csv`.
  - Added optional receiver magnetic-rate mode:
    - `--magnetic-dbdt-mode curl` keeps the original E-form `-curl(E)` output.
    - `--magnetic-dbdt-mode biot_rate` uses the finite-difference rate of the
      Biot-Savart receiver `H` as a diagnostic `dBzdt` path.
  - Validation summaries now distinguish strict component-wise acceptance from
    physical acceptance:
    - `pass_all_components` remains the strict per-component robust-error gate.
    - `physical_pass_all_components` allows weak horizontal electric components
      such as near-zero `Ey` to pass by absolute error scaled by
      `max(|Eh_ref|)`.
    - `weak_components`, `weak_component_passed`, and
      `physical_failed_components` are written to `error_summary.json`.
  - Biot-Savart receiver `Hz` and optional `biot_rate` now use the same
    point/average receiver sampling points as `Ex/Ey` and curl `dBzdt`.

- `dolfinx/legacy_total_field_baseline.py`
  - Frozen copy of the current total-field baseline implementation.

- `src/atem3d/materials/prony.py`
  - `DebyeTerm`
  - `PronyConductivity`
  - Backward-Euler `alpha`, `beta`, and `sigma_eff`.
  - `chi_k` memory initialization/update helpers.
  - `J = sigma_inf E - sum(delta_sigma_k chi_k)` current-density helper.
  - Total-field BE elimination helpers for `sigma_eff E_new`, RHS history current, and eliminated `J_new`.
  - Conversion to/from the existing `atem3d.ip.DebyeIPModel` for solver migration.

- `src/atem3d/primary/`
  - `PrimaryFieldProvider`
  - `ZeroPrimaryProvider`
  - `CachedPrimaryProvider`
  - `EmpymodPrimaryProvider` with no import-time dependency on `empymod`.
  - Injected-runner receiver `E` and `dBdt` primary sampling via `EmpymodSurvey`.

- `src/atem3d/solvers/dc_secondary.py`
  - `DCSecondaryInitialization`
  - `initialize_dc_secondary`
  - Zero-contrast exact `Es0=0` path.
  - IP memory initialization `chi0 = Ep0 + Es0`.
  - Initial secondary current contrast `deltaJ0`.

- `src/atem3d/solvers/tdem_secondary.py`
  - `SecondaryState`
  - `secondary_step_noip`
  - `secondary_step_ip`
  - Variable-`dt` backward-Euler secondary RHS density helpers through injected solvers.
  - Exact zero-contrast secondary response for no-IP and zero-delta IP.

- `src/atem3d/validation_3comp.py`
  - `ThreeComponentValidationInput`
  - `write_three_component_validation_artifacts`
  - Required no-IP/IP validation CSV/JSON/PNG output names.
  - `run_config_resolved.yaml` resolved validation configuration output.
  - Robust relative error and peak-normalized error output through existing metric code.
  - IP Prony metadata in `error_summary.json`.

- `src/atem3d/cli.py`
  - Keeps legacy simulation run mode.
  - Adds `validate-noip-3comp CONFIG.yaml`.
  - Adds `validate-ip-3comp CONFIG.yaml`.
  - Uses lazy imports so validation CLI does not require heavy simulation dependencies.

- `src/atem3d/empymod_validation.py` and `src/atem3d/empymod_validation_cli.py`
  - Lazy-load simulation construction dependencies.
  - Keep empymod validation unit tests runnable with injected runners/reference runners.
  - `atem3d-validate-empymod --artifact-dir ...` can write P7 three-component validation artifacts from an `EmpymodValidationResult`.

- `pyproject.toml`
  - Adds `tdem-ip-forward = "atem3d.cli:main"` command alias.

- `src/atem3d/materials/material_map.py`
  - `CellMaterialMap`
  - `mark_leakage_channel`
  - `apply_leakage_channel_marker`

- `src/atem3d/examples/leakage_channel.py`
  - Synthetic terrain elevation.
  - Irregular leakage-channel marker assignment.
  - Background/no-IP and leakage/IP material map diagnostics.

## Tests Added

- `tests/test_waveform.py`
- `tests/test_time_grid.py`
- `tests/test_error_metric_floor.py`
- `tests/test_dolfinx_validation_artifacts.py`
- `tests/test_dolfinx_biot_receiver.py`
- `tests/test_prony.py`
- `tests/test_primary_provider.py`
- `tests/test_dc_initialization.py`
- `tests/test_secondary_zero_contrast.py`
- `tests/test_noip_3comp_validation_smoke.py`
- `tests/test_ip_3comp_validation_smoke.py`
- `tests/test_complex_terrain_leakage_smoke.py`
- `tests/test_validation_3comp_cli.py`
- Existing `tests/test_empymod_validation.py`
- Existing `tests/test_empymod_validation_cli.py`

## Validation Command

WSL/DOLFINx environment check:

```bash
python dolfinx/sotem_pipeline.py --check-env-only --no-install
```

Observed in WSL `fenicsx` conda environment:

- DOLFINx `0.8.0`
- empymod `2.5.4`
- gmsh `4.15.2`
- pipeline reports regularized volume source fallback unless explicitly overridden.

Lightweight P0-P2 tests:

```bash
python -m pytest -q tests/test_waveform.py tests/test_time_grid.py tests/test_error_metric_floor.py tests/test_dolfinx_validation_artifacts.py tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_solves_through_ramp_then_returns_observation_times tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_keeps_ramp_grid_when_observations_end_before_ramp tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_uses_ramp_solver_t_min_before_later_observation_start
```

P3 material-interface tests:

```bash
python -m pytest -q tests/test_prony.py tests/test_ip_model.py tests/test_debye_fit.py
```

P4 primary-provider tests:

```bash
python -m pytest -q tests/test_primary_provider.py
```

P5 DC secondary initialization tests:

```bash
python -m pytest -q tests/test_dc_initialization.py
```

P6 TDEM secondary stepper tests:

```bash
python -m pytest -q tests/test_secondary_zero_contrast.py
```

P7 no-IP/IP validation smoke tests:

```bash
python -m pytest -q tests/test_noip_3comp_validation_smoke.py tests/test_ip_3comp_validation_smoke.py
```

P7 validation CLI tests:

```bash
python -m pytest -q tests/test_validation_3comp_cli.py
```

Empymod validation helper/CLI tests:

```bash
python -m pytest -q tests/test_empymod_validation.py tests/test_empymod_validation_cli.py
```

Latest P2 receiver/error-summary regression command:

```bash
python -m pytest -q tests/test_dolfinx_validation_artifacts.py tests/test_dolfinx_partial_forward.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_analytic_dc.py tests/test_dolfinx_biot_receiver.py tests/test_error_metric_floor.py tests/test_noip_3comp_validation_smoke.py tests/test_validation_3comp_cli.py tests/test_empymod_validation_cli.py tests/test_empymod_validation.py
```

P8 complex-terrain leakage-channel smoke tests:

```bash
python -m pytest -q tests/test_complex_terrain_leakage_smoke.py
```

## Running No-IP Three-Component Validation

Use the existing DOLFINx pipeline. Example parameters for the current model:

```bash
python dolfinx/sotem_pipeline.py \
  --workdir dolfinx/paper_reproduction/song2025_layered_polarization/paper_y200_center_noip_p2 \
  --source-mode manual_line \
  --source-term-mode impressed_current \
  --source-start-x -500 --source-start-y 200 --source-start-z -0.1 \
  --source-end-x 500 --source-end-y 200 --source-end-z -0.1 \
  --receiver-x 0 --receiver-y -300 --receiver-z -0.1 \
  --source-current 10 \
  --ramp-off-time 1e-5 \
  --t-min 1e-5 --t-max 1 \
  --time-growth 1.25 \
  --min-steps-during-turnoff 10 \
  --outer-boundary-mode robin \
  --outer-boundary-robin-scale 0.1 \
  --magnetic-receiver-mode biot_current \
  --receiver-evaluation-mode median \
  --receiver-diagnostic-types point,disk_average \
  --receiver-average-radius 2 \
  --rho-air 1e6 \
  --rho-earth 100 \
  --layer-depths 350,650 \
  --layer-resistivities 100,100,100 \
  --empymod-srcpts 21
```

The run writes both the legacy combined plot and the new P2 validation artifacts.

## Current Accuracy Status

The current no-IP DOLFINx runs still exceed the 5% target. The corrected
geometry is used throughout:

- Source: `(-500, 200, -0.1) -> (500, 200, -0.1)`.
- Receiver: `(0, -300, -0.1)`.
- Current: `10 A`.
- Time range: `1e-5 s <= t_obs <= 1 s` unless noted.
- Layer model: `rho_air=1e6 ohm m`, earth layers `100 ohm m`, depths
  `350,650 m`.

Small-domain baseline:

- Directory: `dolfinx/current_task_runs/y200_rxminus300_noip_meshcheck`.
- Runtime: total `15.585 s`, forward solve `8.204 s` for the final resumed
  segment.
- Result:
  - `max_error_Ex = 3.951694157397999`
  - `max_error_Ey = 4695844246.711698`
  - `max_error_Hz_or_dBzdt = 0.9083946154727693`
  - `pass_all_components = false`
- Diagnostic: late diffusion length at `1 s` is about `12615.7 m`; the small
  domain is too small.

Large-domain run:

- Directory: `dolfinx/current_task_runs/y200_rxminus300_noip_bigbox_meshcheck`.
- Domain: `x,y = +/-30000 m`, air height `10000 m`, earth depth `30000 m`.
- Mesh preflight: `84952` tetrahedra from mesh-only, estimated memory
  `2.5737 GB`.
- Runtime: total `437.870 s`, forward solve `427.757 s`.
- Result:
  - `max_error_Ex = 0.8363447822943816`
  - `max_error_Ey = 13188019492.654942`
  - `max_error_Hz_or_dBzdt = 1.3215378612016497`
  - `pass_all_components = false`
- Empymod source integration audit `srcpts=21` versus `srcpts=101` shows
  negligible Ex/Hz/dBzdt reference differences, so finite-source integration is
  not the dominant error source.

Large-domain local-refinement run:

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_bigbox_refine80_meshcheck`.
- Local mesh: source/receiver mesh size `80 m`, refinement radius `500 m`.
- Mesh preflight: `98654` tetrahedra, `16948` nodes, estimated memory
  `2.986287 GB`.
- Runtime: total `538.803 s`, forward solve `528.576 s`.
- Result after task-book floor policy (`E_floor=max(1e-14,1e-6*peak)`,
  `H_floor=max(1e-16,1e-6*peak)`, `dBdt_floor=max(1e-18,1e-6*peak)`):
  - `max_error_Ex = 0.3931799684057857`
  - `max_error_Ey = 5150102111.510694`
  - `max_error_Hz_or_dBzdt = 1.190629455795599`
  - `max_peak_normalized_error_Ex = 0.27544871919188463`
  - `max_peak_normalized_error_Hz_or_dBzdt = 0.5297567763762017`
  - `pass_all_components = false`
- Interpretation: local refinement strongly improves early Ex and Hz, but
  dBzdt/curl recovery and late-time robust relative error still fail.

Time-step audit:

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_bigbox_refine80_tg125_t005`.
- Same local-refinement mesh, `time_growth=1.25`, truncated to `t_obs=0.05 s`.
- Runtime: total `1002.950 s`, forward solve `988.441 s`.
- Result:
  - `max_error_Ex = 0.27643232362169967`
  - `max_error_Ey = 5150102111.501422`
  - `max_error_Hz_or_dBzdt = 0.6688288746586054`
  - `max_peak_normalized_error_Ex = 0.27552445469860964`
  - `max_peak_normalized_error_Hz_or_dBzdt = 0.5274395259467757`
  - `pass_all_components = false`
- Interpretation: smaller time growth improves RMS errors but does not solve
  the main discrepancy. It also raises late KSP iteration counts substantially.

Current root-cause status:

- Boundary size is a real contributor: Ex RMS improves from the small-domain
  run to the big-box runs.
- Empymod finite-source integration is not the main cause.
- Local source/receiver mesh refinement is a major contributor: early Ex
  improves from about `83.6%` max error to about `27.6%` in the first-five-point
  comparison.
- Time-step density is a secondary contributor.
- Remaining dominant issues are likely receiver sampling/curl recovery near the
  shallow interface, local near-source/receiver discretization, and possibly the
  total-field source transfer into the E-form solve.

Receiver averaging smoke:

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_diskavg_receiver_smoke`.
- Command used `--receiver-type disk_average --receiver-average-radius 2` with
  `--stop-after-outputs 1`.
- Result: DOLFINx completed the first observation point and wrote the standard
  validation artifacts. The first receiver row was
  `Ex=6.467738e-04`, `Ey=-5.793007e-05`, `Hz=-2.111455e-03`,
  `dBzdt=2.768666e-06`.
- Runtime: total `77.323 s`, forward solve `63.619 s` in the WSL `fenicsx`
  environment.
- Interpretation: average receiver sampling is now a working diagnostic path,
  and the pipeline can now output simultaneous point/average diagnostic curves
  through `receiver_diagnostics.csv`. A new WSL validation run is still needed
  to populate those simultaneous diagnostic rows for the latest large-domain
  model.

Simultaneous point/disk receiver diagnostic smoke:

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_point_diskdiag_smoke`.
- Command used the corrected latest model:
  `(-500, 200, -0.1) -> (500, 200, -0.1)`, receiver `(0, -300, -0.1)`,
  `10 A`, `rho_air=1e6 ohm m`, earth layers `100 ohm m` with depths
  `350,650 m`, big box `x,y=+/-30000 m`, `air_height=10000 m`,
  `earth_depth=30000 m`, local source/receiver mesh size `80 m`, radius
  `500 m`.
- Receiver diagnostics used
  `--receiver-type point --receiver-diagnostic-types point,disk_average
  --receiver-average-radius 2`.
- WSL run was split by checkpoint:
  - First segment: one output point, terminal wall time `92.8 s`.
  - Resume segment: four additional output points, terminal wall time `49.0 s`.
  - Resume report runtime: total `41.017 s`, forward solve `27.619 s`,
    empymod reference `6.306 s`.
- Artifacts include:
  - `receiver_diagnostics.csv`
  - `receiver_diagnostics.png`
  - `receiver_reference_errors.csv`
  - `receiver_reference_error_curves.png`
  - standard `comparison_3comp.png`, `error_curves_3comp.png`,
    `predictions.csv`, `reference_empymod_or_1d.csv`, `errors.csv`,
    `error_summary.json`, `diagnostics.json`.
  - `diagnostics.json` now includes a `receiver_sampling` summary with
    point/disk component-wise relative differences and
    `receiver_sampling_issue_suspected=true`.
  - `diagnostics.json` also records source charge-conservation projection
    residuals: before `5.663467`, after `2.869899e-10`,
    endpoint norm `1.414214`.
  - `diagnostics.json` includes `magnetic_recovery` from a Faraday-integrated
    `dBzdt` receiver trace. Over the first five points, the integrated `Hz`
    differs from reported Biot `Hz` by at most about `0.45%`.
  - `magnetic_recovery.rate_consistency` now compares `mu0*dHz/dt` against
    the trapezoid-averaged `dBzdt` over each output interval. For this smoke
    run the maximum interval relative difference is about `42.2%` at
    `t_mid=1.125e-5 s`, showing that the short-window integrated-Hz diagnostic
    can hide a substantial derivative-level inconsistency.
  - `diagnostics.json` includes `receiver_vs_reference`, and the same
    diagnostic receiver/reference errors are now emitted as the automatic CSV
    and PNG artifacts listed above. `disk_average` improves `dBzdt` max error
    from about `53.1%` to `38.3%`, but still fails the `5%` target. It does
    not improve `Ex` or weak `Ey`.
- Result over the first five output times
  (`1.0e-5` to `2.44140625e-5 s`):
  - `max_error_Ex = 0.27639692569161395`
  - `max_error_Ey = 5150102111.501422`
  - `max_error_Hz_or_dBzdt = 0.5308318716061439`
  - `pass_all_components = false`
- Point/disk receiver diagnostic difference from the same field solve:
  - `Ex`: about `0.19%` to `2.35%`.
  - `Ey`: about `12.48%` to `21.82%`; this remains a near-zero component.
  - `dBzdt`: about `9.15%` to `31.49%`.
- Against empymod, `disk_average` reduces `dBzdt` max error from `53.1%`
  to `38.3%`, but the response is still far above the `5%` gate.
- Interpretation: simultaneous receiver diagnostics now work. The disk-average
  electric response is close to the point response for `Ex`, but `dBzdt`
  changes substantially, supporting the current diagnosis that receiver/curl
  recovery near the shallow interface is a major contributor to the early
  magnetic-response error. The Faraday-integrated `Hz` diagnostic is close to
  the Biot `Hz` trace over this short window, so the current evidence points
  more strongly at `dBzdt`/curl recovery than at Biot `Hz` recovery.

Biot-rate dBdt diagnostic smoke:

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_biotrate_smoke`.
- Command used the same corrected latest-model geometry and mesh settings as
  the point/disk smoke, but added `--magnetic-dbdt-mode biot_rate`.
- The run completed one output point in WSL and WSL was shut down afterwards.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.2765931106438829`
  - `max_error_Ey = 5150102111.510694`
  - `max_error_Hz_or_dBzdt = 0.3754687533711432`
  - `pass_all_components = false`
- Interpretation: deriving `dBzdt` from the Biot-Savart receiver `Hz` rate
  improves the first point relative to point curl recovery
  (`~53.1% -> ~37.5%`) and is comparable to disk-average curl recovery
  (`~38.3%`). This confirms that receiver/curl recovery is a major error
  channel, but this branch still does not meet the `5%` gate.

Receiver mesh-sensitivity smoke:

- Summary CSV:
  `dolfinx/current_task_runs/y200_rxminus300_receiver_mesh_sensitivity.csv`.
- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_recv20_biotrate_smoke`.
- Change relative to the 80 m local-refinement baseline: receiver mesh size
  `20 m`, receiver refinement radius `200 m`, source mesh size still `80 m`,
  `--magnetic-dbdt-mode biot_rate`, one output point.
- Mesh preflight: `132000` cells in the `.msh` file, estimated memory
  `3.7939785 GB`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.13874094959702565`
  - `max_error_Ey = 1601943911.790958`
  - `max_error_Hz_or_dBzdt = 0.44747549498846956`
  - `pass_all_components = false`
- Interpretation: receiver-local refinement improves the first-point `Ex`
  error from about `27.6%` to about `13.9%`, and `Hz` is essentially
  matched (`3.6e-5` relative error). However, the Biot-rate `dBzdt` remains
  far above `5%`.
- Candidate-cell diagnostic from the rerun:
  - point receiver: `sample_count=1`, `candidate_count=32`.
  - disk receiver: `sample_count=5`, `candidate_count_min=4`,
    `candidate_count_max=32`, `candidate_count_mean=9.6`.
  This confirms that near-interface receiver extraction is sampling many
  colliding cell candidates, so cell-selection policy is not a minor detail.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_recv20_nearest_biotrate_smoke`.
- Change: same receiver 20 m mesh as above, but
  `--receiver-evaluation-mode nearest_center`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.13800838032493573`
  - `max_error_Ey = 1268457408.674592`
  - `max_error_Hz_or_dBzdt = 0.44747549498846956`
  - `pass_all_components = false`
- Interpretation: nearest-center candidate selection only slightly improves
  point `Ex` relative to median (`13.9% -> 13.8%`). The simultaneous
  disk diagnostic improves `dBzdt` to about `14.9%`, but still does not reach
  `5%`. This suggests that the remaining first-point error is not caused only
  by arbitrary candidate-cell ordering; local source/receiver discretization
  and the shallow-interface total-field transfer still need improvement.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src40_recv20_nearest_biotrate_smoke`.
- Change: source mesh size `40 m`, source refinement radius `300 m`,
  receiver mesh size `20 m`, `nearest_center`, one output point.
- Mesh preflight: `142161` cells in the `.msh` file, estimated memory
  `4.0858845 GB`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.11406489203543935`
  - `max_error_Ey = 5952295267.625508`
  - `max_error_Hz_or_dBzdt = 0.15683277651412947`
  - `pass_all_components = false`
- Diagnostic receiver comparison from the same solve:
  - point `dBzdt` error: about `15.7%`.
  - disk-average `dBzdt` error: about `3.0%` and passes the first-point
    `5%` gate.
- Interpretation: source-local refinement changes the early response
  substantially. Compared with the source80/receiver20 run, `Ex` moves from
  under-prediction to over-prediction, while disk/curl `dBzdt` becomes
  acceptable for the first point. This makes the source transfer/local source
  mesh a major remaining P2 error channel.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src40_recv20_nearest_biotrate_localdiag_smoke`.
- Change: same source40/receiver20/nearest/biot-rate smoke as above, with
  additional manual-line local source projection diagnostics in
  `diagnostics.json` and `verification_report.txt`.
- Source local projection diagnostics:
  - quadrature points: `501`.
  - added points: `501`.
  - missed points: `0`.
  - unique hit cells: `202`.
  - max hit-cell fraction: `0.043912175648702596`.
  - top cell L1 contribution fraction: `0.011818270030426044`.
  - top DOF L1 contribution fraction: `0.011150985042165202`.
  - endpoint windows: start/end both have `72` points, `10` unique cells,
    and `0` missed points.
- Result at `t_obs=1.0e-5 s` remains unchanged:
  - `max_error_Ex = 0.11406489203543935`
  - `max_error_Ey = 5952295267.625508`
  - `max_error_Hz_or_dBzdt = 0.15683277651412947`
  - `pass_all_components = false`
- Interpretation: this rules out an obvious local-projection concentration
  failure for the source40 manual-line run: the source line is fully hit,
  distributed over many cells, and no endpoint-window misses occur. The
  remaining P2 first-point error is more likely in total-field source/DC
  coupling, shallow-interface transfer, receiver/curl recovery, or boundary
  treatment than in a single missing or dominant source cell.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src40_recv20_nearest_biotrate_projectiondiag_smoke`.
- Change: same source40/receiver20/nearest/biot-rate smoke as above, with
  source charge-conservation projection correction norms added to
  `source_projection`.
- Source projection correction diagnostics:
  - raw source L2 norm: `14.61846112295302`.
  - projected source L2 norm: `14.403726221660904`.
  - correction L2 norm: `2.5000617875663655`.
  - correction L2/raw: `0.1710208596198214`.
  - raw source L1 norm: `201.65139957200677`.
  - projected source L1 norm: `244.89689428028413`.
  - correction L1 norm: `62.72421942125531`.
  - correction L1/raw: `0.3110527353362475`.
  - divergence residual reduction: `0.9999999993141232`.
- Result at `t_obs=1.0e-5 s` remains:
  - `max_error_Ex = 0.11406489203543935`
  - `max_error_Ey = 5952295267.625508`
  - `max_error_Hz_or_dBzdt = 0.15683277651412947`
  - `pass_all_components = false`
- Interpretation: the charge-conservation projection is numerically effective
  for endpoint balance, but it changes the assembled source vector by a
  non-negligible amount (`17%` in L2 and `31%` in L1). The next diagnostic
  should compare a raw-source run against the charge-conserving run at the
  first output point to determine whether this projection is a main driver of
  the remaining `Ex` discrepancy or only a necessary consistency correction.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src40_recv20_nearest_biotrate_rawproj_smoke`.
- Change: same source40/receiver20/nearest/biot-rate smoke as above, but
  with `--source-projection-mode raw`, so the assembled manual-line vector is
  used without the endpoint charge-conservation projection.
- Source projection diagnostics:
  - projection mode: `raw`.
  - applied: `false`.
  - before residual: `6.561607663609301`.
  - after residual: `6.561607663609301`.
  - endpoint norm: `1.414213562373099`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.04866187758167016`
  - `max_error_Ey = 1033376377.0488973`
  - `max_error_Hz_or_dBzdt = 0.17328421255717028`
  - `pass_all_components = false`
- Diagnostic receiver comparison from the same solve:
  - disk-average `Ex` error: about `3.91%`.
  - disk-average `dBzdt` error: about `2.66%`.
- Interpretation: disabling the projection pulls first-point `Ex` below the
  `5%` gate and keeps disk-average `dBzdt` below `5%`, but this raw source
  violates the endpoint balance by the full `6.56` residual and worsens point
  `dBzdt`. This strongly indicates that the projection/source consistency
  pathway is a main contributor to the `Ex` discrepancy, but raw mode is not a
  physically acceptable final solver. The next implementation step should
  search for a charge-conserving source construction that preserves the
  physically correct near-source moment, instead of using the current global
  gradient projection as the only correction.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src40_recv20_nearest_biotrate_rawproj_balance_smoke`.
- Change: same raw-source diagnostic run, with scalar source-balance vector
  distribution diagnostics added under `source_projection.scalar_balance`.
- Scalar balance diagnostics before projection:
  - endpoint active scalar DOFs: `2`.
  - raw current-divergence active scalar DOFs: `216`.
  - residual active scalar DOFs: `216`.
  - endpoint L2 norm: `1.414213562373099`.
  - raw current-divergence L2 norm: `6.703242996163269`.
  - residual L2 norm: `6.561607663609301`.
  - residual L2 / endpoint L2: `4.639757274423743`.
  - residual Linf norm: `2.2542275461651973`.
  - residual top-absolute fraction: `0.03433153794676107`.
  - current-divergence/endpoint alignment: `0.20458050351679713`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.04866187758167016`
  - `max_error_Ey = 1033376377.0488973`
  - `max_error_Hz_or_dBzdt = 0.17328421255717028`
  - `pass_all_components = false`
- Interpretation: the raw source-balance residual is distributed over many
  scalar DOFs, not concentrated at the two endpoints. A simple local endpoint
  patch is therefore unlikely to be sufficient. The more likely fault is that
  the direct manual-line Nedelec integral is not fully compatible with
  DOLFINx's discrete-gradient orientation/transformation convention on the
  unstructured tetrahedral mesh. The next step should audit the source vector
  against explicit `G.T @ s` contributions cell-by-cell or switch to a
  DOLFINx-native line/source assembly that preserves the de Rham identity.

- Direct source-only quadrature sweep:
  - `source_only_y200_rxminus300_src40_recv20_raw_qauto` used the old automatic
    `501` Gauss points and had source-balance residual `6.561607663609301`.
  - `source_only_y200_rxminus300_src40_recv20_raw_q2001` used `2001` Gauss
    points and reduced the residual to `1.6073351550413124`.
  - `source_only_y200_rxminus300_src40_recv20_raw_q5001` used `5001` Gauss
    points and reduced the residual to `0.5548956230007095`.
  - After this diagnostic, the automatic manual-line quadrature rule was
    changed to resolve the source mesh scale more tightly; for the corrected
    latest model it now chooses `5001` points without requiring a manual
    `--source-quadrature-points` override.
- Interpretation: the large raw-source residual is mainly a line-integration
  resolution problem across a discontinuous cellwise integrand, not only an
  orientation/sign issue. Exact cell-segmented integration is still preferable,
  but dense automatic quadrature sharply reduces the required projection
  correction while staying within the 32 GB memory budget.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src40_recv20_nearest_biotrate_q5001_smoke`.
- Change: same source40/receiver20/nearest/biot-rate smoke as above, but with
  dense `5001`-point manual-line quadrature and charge-conserving projection.
- Source projection diagnostics:
  - before residual: `0.5548956230007095`.
  - after residual: `5.351519776153429e-10`.
  - correction L2/raw: `0.015237824962516272`.
  - correction L1/raw: `0.02394478572405994`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.03385766465143364`
  - `max_error_Ey = 1902358644.1520596`
  - `max_error_Hz_or_dBzdt = 0.1735791517790405`
  - `pass_all_components = false`
- Diagnostic receiver comparison from the same solve:
  - disk-average `Ex` error: `0.023888841807233482`.
  - disk-average `dBzdt` error: `0.02873626890433773`.
- Interpretation: dense quadrature plus charge-conserving projection brings
  first-point `Ex` below the `5%` gate and makes disk-average `Ex/dBzdt`
  pass. The remaining first-point failure is now mainly the point `dBzdt`
  recovery (`~17.36%`) and near-zero `Ey`, not the horizontal source electric
  amplitude. This is a significant P2 improvement, but it is still not a full
  three-component `5%` pass.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_diskavg_curl_q5001_weakgate_smoke`.
- Change: same corrected latest model, dense automatic `5001`-point source
  quadrature, charge-conserving source projection, main receiver
  `disk_average`, `--magnetic-receiver-mode biot_current`, and
  `--magnetic-dbdt-mode curl` so `dBzdt` follows the E-form `-curl(E)` path.
- Runtime: total `114.865 s`, mesh `14.313 s`, setup `11.738 s`, forward solve
  `83.496 s`, empymod reference `1.474 s`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.023888841807233482`
  - `max_error_Ey = 3026120491.2908845`
  - `max_error_Hz_or_dBzdt = 0.02873626890433773`
  - `weak_component_scaled_abs_error_max(Ey) = 0.03378317518993349`
  - `pass_all_components = false`
  - `physical_pass_all_components = true`
- Interpretation: for the first observation point, the physically meaningful
  P2 gate passes: `Ex`, curl `dBzdt`, `Hz`, `Eh_vector`, and weak-`Ey` scaled
  absolute error are all below `5%`. The strict row-wise component gate remains
  false because ordinary/robust scalar relative error on near-zero `Ey` is
  ill-conditioned. This is not yet a full `1e-5 s` to `1 s` acceptance result;
  it is a first-point corrected-model smoke result.

- Full-window continuation for the same directory was completed to
  `t_obs=1.0 s` from the saved checkpoint.
- The first resume using the default `--max-it 1000` reached
  `t_obs=0.2869859254937225 s` but stopped at the iteration cap after about
  `1942.5 s` wall time. This was a linear-solver iteration limit, not a memory
  failure.
- A second resume with `--max-it 3000` completed the remaining window to
  `1.0 s`. The final postprocessed report records 53 observation times and a
  postprocess-only runtime of `6.437 s`; the final resume segment took about
  `100.9 s` wall time in WSL.
- Full-window result:
  - `pass_all_components = false`
  - `physical_pass_all_components = false`
  - `physical_failed_components = ["Ex", "Hz", "dBzdt"]`
  - `weak_components = ["Ey"]`
  - `weak_component_passed = true`
  - `weak_component_scaled_abs_error_max(Ey) = 0.04275824798239449`
  - `max_error_Ex = 0.2976303365126635` at
    `t_obs=0.03081487911019577 s`
  - `max_error_Hz = 0.18380782675248264` at
    `t_obs=0.012621774483536187 s`
  - `max_error_dBzdt = 0.77079512254321` at
    `t_obs=0.019721522630525293 s`
  - `max_peak_normalized_error_Ex = 0.09545122731533442`
  - `max_peak_normalized_error_Hz = 0.05201560660020918`
  - `max_peak_normalized_error_dBzdt = 0.029014850706464392`
- Receiver diagnostics over the 53-time full window:
  - `disk_average` vs `point` `Ex` max relative difference:
    `0.057614147566712`
  - `disk_average` vs `point` `dBzdt` max relative difference:
    `0.02935319014773`
  - `disk_average` slightly improves reference error for `dBzdt` but does not
    bring it below the `5%` robust-relative gate.
- Magnetic recovery diagnostics:
  - Faraday-integrated `Hz` from curl `dBzdt` ends at
    `1.4787901570827183e-04`.
  - Reported Biot-Savart `Hz` ends at `-2.0103580162363813e-08`.
  - `max_relative_hz_difference = 7356.854753926774` at `t_obs=1.0 s`.
  - This supports treating Biot-Savart `Hz` as an auxiliary diagnostic over the
    full window; E-form validation should prioritize curl `dBzdt` until a
    consistent Faraday-integrated `Hz` receiver is implemented.
- Boundary/refinement diagnostic after report fix:
  - `Lmax(t_max) = 12615.7 m`
  - recommended radius/depth `>= 25231.3 m`
  - finite domain radius/depth are both `30000 m`, so
    `domain_underresolved = false`
  - the separate late-diffusion local refinement box remains
    `1000 m x 500 m`, so `refinement_underresolved = true`
- Interpretation: the latest corrected no-IP total-field run now covers the
  required `1e-5 s <= t_obs <= 1 s` window and writes all required P2
  artifacts, but it does not meet the task-book `5%` physical gate. The weak
  `Ey` treatment is no longer the main issue. The remaining failures are
  dominated by mid-time `Ex`, magnetic recovery/`Hz`, and scalar robust
  relative `dBzdt` when the reference has decayed, despite low
  peak-normalized `dBzdt` error.

- Mesh-segment source integration update:
  - Code now reads Gmsh `source_wire` line elements from `verification_mesh.msh`
    when available and uses them as source-line integration segments.
  - Each line segment is still sampled densely according to the existing
    automatic target spacing; this avoids the failed two-point-per-segment
    experiment where collision-cell selection on embedded line entities was too
    ambiguous.
  - If source line elements are unavailable, the solver falls back to the
    previous global Gauss rule.
  - Report output now records `source line integration: mode=...`,
    segment count, total segment length, and quadrature-per-segment summary.
- Source-only smoke:
  - Directory:
    `dolfinx/current_task_runs/source_only_y200_rxminus300_src40_recv20_meshseg`.
  - Corrected latest model, source mesh size `40 m`, receiver mesh size `20 m`.
  - Mesh line segments: `200`, total segment length `1000 m`.
  - Adaptive segment quadrature: `5200` points, `26` points per segment.
  - Missed source quadrature points: `0`.
  - Source projection before residual: `0.15869511184790644`.
  - Source projection after residual: `2.1986488276977117e-10`.
  - Projection correction `L2/raw = 0.0053012314244093054`,
    improved from the previous dense-global-q5001 value of about `0.01524`.
- First-output mesh-segment no-IP smoke:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_src40_recv20_diskcurl_smoke`.
  - Main receiver: `disk_average`; magnetic validation quantity: curl `dBzdt`.
  - Result at `t_obs=1.0e-5 s`:
    - `max_error_Ex = 0.02878542212309011`
    - `max_error_Hz = 0.001107642062978224`
    - `max_error_dBzdt = 0.025925605699880186`
    - weak `Ey` scaled absolute error: `0.014975738473896322`
    - `physical_pass_all_components = true`
  - Point receiver from the same field also passes `Ex` and `dBzdt` at the
    first output: point `Ex` error `0.03799459110341777`, point `dBzdt` error
    `0.04882181606313593`.
- Five-output mesh-segment no-IP smoke:
  - Same directory as above, regenerated with `--stop-after-outputs 5`.
  - Runtime: total `143.321 s`, mesh `14.294 s`.
  - Time range: `1.0e-5` to `2.44140625e-5 s`.
  - Source line integration:
    `mode=mesh_segments`, `segments=200`, total segment length `1000 m`,
    `26` quadrature points per segment, `5200` total quadrature points.
  - Source projection:
    - before residual: `0.15869511184790644`
    - after residual: `2.1986488276977117e-10`
    - correction `L2/raw = 0.0053012314244093054`
  - Five-output errors:
    - `max_error_Ex = 0.09418477867481403` at
      `t_obs=2.44140625e-5 s`
    - `max_error_Hz = 0.00520392194214342`
    - `max_error_dBzdt = 0.026648609495953234`
    - weak `Ey` scaled absolute error: `0.014953312810585524`
    - `physical_pass_all_components = false`
    - `physical_failed_components = ["Ex"]`
  - Comparison with the earlier dense-global-q5001 disk-average run over the
    same five times:
    - q5001 `Ex` errors were approximately
      `2.42%, 5.48%, 7.33%, 8.46%, 9.11%`.
    - mesh-segment `Ex` reaches `9.42%` at the fifth point, so source balance
      improved but early `Ex` did not improve enough.
    - mesh-segment `dBzdt` remains below `2.67%` and is comparable to or
      slightly better than q5001 over the same early window.
  - Interpretation: mesh-segment adaptive source integration is a real P2
    improvement for source consistency and magnetic early-time response, but
    it does not fix early-time `Ex`. A full `1 s` rerun is not justified until
    the early `Ex` channel is addressed.
- Ramp-off time-step sensitivity smoke:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_src40_recv20_diskcurl_ramp20_smoke`.
  - Change relative to the five-output mesh-segment run:
    `min_steps_during_turnoff=20` and `ramp_solver_t_min=5e-7 s`.
  - Runtime: total `221.268 s`, mesh `14.343 s`.
  - Time range: `1.0e-5` to `2.44140625e-5 s`.
  - Result:
    - `max_error_Ex = 0.09406775301612384`
    - `max_error_Hz = 0.004656033383964945`
    - `max_error_dBzdt = 0.027039912833121443`
    - weak `Ey` scaled absolute error: `0.015023585990404906`
    - `physical_pass_all_components = false`
    - `physical_failed_components = ["Ex"]`
  - Comparison with ramp10:
    - ramp10 `max_error_Ex = 0.09418477867481403`
    - ramp20 `max_error_Ex = 0.09406775301612384`
    - Difference is negligible relative to the `5%` gate.
  - Interpretation: doubling the ramp-off internal steps does not fix early
    `Ex`. The remaining error is unlikely to be dominated by turn-off-history
    time discretization; next diagnostics should focus on initial DC/source
    coupling, E receiver extraction near the shallow interface, or total-field
    source transfer.

- Analytic DC initial-field smoke:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_src40_recv20_diskcurl_analyticdc_smoke`.
  - Change relative to the five-output mesh-segment run:
    `--initial-dc-mode analytic_halfspace`.
  - Result over the first five output times:
    - `max_error_Ex = 0.06799536786261284`
    - `max_error_Hz = 0.009233751670207124`
    - `max_error_dBzdt = 0.02749341398150946`
    - weak `Ey` scaled absolute error: `0.033933136602542754`
    - `physical_pass_all_components = false`
    - `physical_failed_components = ["Ex"]`
  - Interpretation: analytic halfspace DC improves `Ex` compared with the FEM
    DC initialization, and points 2-5 pass the physical gate, but the first
    after-ramp output still fails. This isolates the remaining early `Ex`
    issue to either the first after-ramp time step or source/DC coupling at
    the turn-off transition.

- After-ramp first-observation substep smoke:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_src40_recv20_diskcurl_analyticdc_substep4_smoke`.
  - Change relative to the analytic DC smoke:
    `--min-steps-before-first-observation 4`, which inserts internal solve
    times `1.25e-5`, `1.50e-5`, and `1.75e-5 s` between `t_off=1.0e-5 s`
    and the first output at `t_internal=2.0e-5 s`.
  - Runtime: total `166.687 s`.
  - Result over the first five output times:
    - `max_error_Ex = 0.02917101885961845`
    - `max_error_Hz = 0.009205488588706368`
    - `max_error_dBzdt = 0.03013405182182564`
    - `max_error_Hz_or_dBzdt = 0.03013405182182564`
    - weak `Ey` scaled absolute error: `0.03382239075406781`
    - `physical_pass_all_components = true`
    - `physical_failed_components = []`
  - Receiver/reference diagnostic:
    - main disk-average `Ex` passes with max error `2.92%`.
    - main disk-average curl `dBzdt` passes with max error `3.01%`.
    - point receiver `Ex` also passes with max error `2.10%`, while point
      `dBzdt` is just above the gate at `5.16%`.
    - `Ey` remains a symmetry-near-zero component; strict scalar relative
      error is not meaningful, but the weak-component scaled absolute gate
      passes.
  - Interpretation: the first post-ramp internal step was a real early-time
    error channel. With analytic DC plus four internal steps before the first
    observation, the corrected latest no-IP total-field model passes the
    physical gate for the first five output times. This is still only an early
    smoke result, not a full `1e-5 s` to `1 s` acceptance result.

- Full-window analytic DC plus first-observation-substep run:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_src40_recv20_diskcurl_analyticdc_substep4_full`.
  - Configuration: same as the five-output substep smoke, but run to
    `t_obs=1.0 s`.
  - Execution:
    - First WSL segment with `--max-it 3000` reached step `50`
      (`t_obs=0.03851859888774471 s`) and then failed at step `51`
      (`t_internal=4.815825e-02 s`) with KSP `reason=-3`.
    - Resume with `--max-it 6000` completed to `t_obs=1.0 s`.
    - WSL was shut down after both segments.
  - Full-window result:
    - `pass_all_components = false`
    - `physical_pass_all_components = false`
    - `physical_failed_components = ["Ex", "Ey", "Hz", "dBzdt"]`
    - `weak_component_passed = false`
    - `weak_component_scaled_abs_error_max(Ey) = 0.0869621825168415`
    - `max_error_Ex = 5501.311526120881` at `t_obs=1.0 s`
    - `max_peak_normalized_error_Ex = 0.129011302486824`
    - `max_error_Hz = 2.8210700979882883` at `t_obs=1.0 s`
    - `max_error_dBzdt = 0.7984781792920356` at
      `t_obs=0.019721522630525293 s`
    - `max_peak_normalized_error_dBzdt = 0.03155282305894918`
  - Diagnostic observation:
    - The early five-output window still passes the physical gate.
    - After about `0.02-0.05 s`, the predicted electric field develops a
      persistent static-like offset. At `t_obs=1.0 s`, predicted `Ex` is
      `-1.1588946989865485e-4 V/m` while empymod reference `Ex` is
      `2.1069619302160945e-8 V/m`.
    - This failure is therefore not the same early first-step error fixed by
      `min_steps_before_first_observation`; it is a later total-field
      gradient/static residual problem.

- Conductivity divergence-cleaning diagnostic:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_substep4_divclean_growth2_diag`.
  - Change relative to the full-window substep run:
    `--divergence-cleaning conductivity` and a coarse diagnostic
    `--time-growth 2.0`.
  - Execution:
    - First segment timed out after 30 minutes at `t_obs=0.16384 s`; WSL was
      then manually shut down.
    - Resume completed to `t_obs=1.0 s`; WSL was shut down afterwards.
  - Result:
    - `pass_all_components = false`
    - `physical_pass_all_components = false`
    - `physical_failed_components = ["Ex", "Hz", "dBzdt"]`
    - `weak_component_passed = true`
    - `weak_component_scaled_abs_error_max(Ey) = 0.02375229937232657`
    - `max_error_Ex = 0.9497349158671408` at `t_obs=0.04096 s`
    - `max_peak_normalized_error_Ex = 0.12733470951899298`
    - `max_error_dBzdt = 3.6629916640217104` at `t_obs=0.04096 s`
    - `max_peak_normalized_error_dBzdt = 0.08630761754389761`
  - Diagnostic observation:
    - The cleaner removes the late static offset: at `t_obs=1.0 s`, predicted
      `Ex` drops to `2.9826907527504154e-8 V/m`, close in absolute scale to
      the empymod reference `2.1069619302160945e-8 V/m`.
    - It is not a valid final fix as currently applied: it over-amplifies or
      distorts the mid-time relative response around `0.02-0.04 s`, especially
      `dBzdt`.
    - Root-cause direction is now clearer: the total-field E-form solve is
      carrying a post-ramp conductivity-divergence/gradient residual. A final
      solution should control that residual without removing physically needed
      inductive response.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_diskavg_biotrate_q5001_weakgate_smoke`.
- Change: same as above, but `--magnetic-dbdt-mode biot_rate`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.023888841807233482`
  - `max_error_Hz_or_dBzdt = 0.1735791517790405`
  - `weak_component_scaled_abs_error_max(Ey) = 0.03378317518993349`
  - `physical_pass_all_components = false`
- Interpretation: `biot_rate` remains a diagnostic magnetic-rate path and does
  not pass the first-point `dBzdt` gate. The preferred P2 validation quantity
  for E-form remains curl `dBzdt`.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_src60_recv20_diskcurl_smoke`.
- Change: source mesh size `60 m`, source refinement radius `400 m`,
  receiver mesh size `20 m`, main receiver `disk_average`, one output point.
- Mesh preflight: `128553` cells in the `.msh` file, estimated memory
  `3.6948645 GB`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.43962372084123774`
  - `max_error_Ey = 26621862371.412643`
  - `max_error_Hz_or_dBzdt = 0.057429855867726115`
  - `pass_all_components = false`
- Interpretation: the source60/source-radius400 configuration is not a
  useful midpoint; it strongly over-predicts `Ex`. The source mesh response is
  non-monotonic, likely because the gmsh source-line embedding/local sizing
  changes the Nedelec line-source projection and nearby shallow-interface
  cells. The next improvement should make source projection diagnostics local
  and geometric, not just global residuals.

- Directory:
  `dolfinx/current_task_runs/y200_rxminus300_noip_recv10_diskcurl_smoke`.
- Change: receiver mesh size `10 m`, receiver refinement radius `120 m`,
  `--receiver-type disk_average`, `--magnetic-dbdt-mode curl`, one output
  point.
- Mesh preflight: `155751` cells in the `.msh` file, estimated memory
  `4.4764695 GB`.
- Result at `t_obs=1.0e-5 s`:
  - `max_error_Ex = 0.2353157835715751`
  - `max_error_Ey = 10634476606.208687`
  - `max_error_Hz_or_dBzdt = 0.19573457682633436`
  - `pass_all_components = false`
- Interpretation: this configuration improves disk/curl `dBzdt` to about
  `19.6%`, but `Ex` worsens relative to the 20 m point case. The improvement
  is not monotonic with receiver mesh size because the average receiver
  samples different cell candidates near the shallow interface. The next
  numerical step should separate point Ex interpolation, disk radius, and cell
  candidate selection before spending a long run to 1 s.

This implementation round improves time-axis correctness and reporting/diagnostics. It does not resolve the known near-source source-transfer/MMR consistency problem or achieve the final 5% no-IP/IP target.

## Known Limitations

- `diagnose_source_consistency` currently reports waveform-integral and endpoint-total checks without full FEM matrix residuals unless a source projection residual is provided.
- `source_projection_mode=raw` is implemented only as a diagnostic switch.
  It can improve first-point `Ex`, but it violates endpoint charge
  conservation and must not be used as a final accepted solver mode.
- Manual-line automatic quadrature was made denser and now fixes the first
  point `Ex` error for the corrected latest model, but it is still an
  approximate quadrature over a discontinuous cellwise integrand.
- Mesh-segment adaptive source integration now uses the Gmsh source-wire line
  elements to split the line integral and improves source balance/first-output
  error. It still uses collision-cell selection for quadrature points on an
  embedded curve, so a stricter de Rham-compatible source assembly or exact
  source-edge orientation audit remains a long-term improvement.
- Average receiver sampling and simultaneous point/average diagnostic CSV/PNG
  artifact output are implemented and smoke-tested for the E-form DOLFINx
  verification pipeline. Biot-Savart `Hz` now honors average receiver sampling.
  H-form diagnostic output still writes only the main receiver response.
- Faraday-integrated `Hz` recovery is not implemented in this round.
- Late-diffusion diagnostics now report finite-domain coverage separately from
  the local refinement box. Existing artifacts generated before this fix may
  still show the old ambiguous `actual radius/depth` wording until
  regenerated with `--postprocess-partial`.
- P3 currently provides the material API and memory-update tests; DOLFINx total-field IP assembly still needs to be migrated to this API and verified against no-IP when `delta_sigma=0`.
- P4 currently provides zero/cached primary providers and receiver-side empymod primary sampling through an injected/reference runner; FEM-space primary field interpolation remains pending.
- P5 currently provides a pure initialization core with an injected secondary field solver; DOLFINx scalar Poisson assembly for `phi_s` remains pending.
- P6 currently provides pure no-IP/IP time-step kernels with injected secondary solvers; DOLFINx curl-curl/mass/Robin operator assembly remains pending.
- P7 currently verifies artifact generation from supplied arrays; it does not yet run a real empymod/1D backend to prove 5% physical agreement.
- P7 CLI currently reads precomputed prediction/reference CSV files; it does not yet launch DOLFINx or empymod itself.
- `atem3d-validate-empymod --artifact-dir` bridges real validation results to artifact files, but final 5% agreement still depends on the underlying simulation/reference result.
- P8 currently verifies marker/material/channel geometry utilities; it does not yet run a DOLFINx gmsh complex-terrain forward example.
- Full no-IP/IP `1e-5 s` to `1 s` 5% acceptance is not yet achieved. The latest
  full-window corrected-model no-IP run with analytic DC and
  `min_steps_before_first_observation=4` covers the full window, but the
  physical gate still fails for `Ex`, `Ey`, `Hz`, and `dBzdt`.
- The latest analytic-DC plus first-observation-substep smoke passes the
  physical no-IP gate over the first five output times only. Full-window
  results show a later post-ramp static/gradient residual in the total-field
  E-form solve.
- Conductivity divergence cleaning removes the late static electric-field
  offset in a coarse diagnostic run, but the current projection changes the
  mid-time amplitude and does not meet the `5%` gate. It is evidence for the
  root cause, not a final accepted solver mode.
- `compute_error` and the validation artifact writer now share the task-book
  floor policy. Older reports generated before this change should be
  regenerated with `--postprocess-partial` before comparing error numbers.

## Next Steps

1. Replace the current all-or-nothing post-ramp conductivity divergence
   projection with a controlled consistency treatment: quantify
   `G^T M_sigma E`, `curl(E)`, receiver amplitude, and reference error before
   and after cleaning, then apply the smallest correction that removes the
   static residual without suppressing mid-time inductive response.
2. Keep mesh-segment line-source integration as the current source baseline,
   and add an explicit de Rham/source-edge orientation audit before replacing
   it with any DOLFINx-native source assembly.
3. Add Faraday-integrated magnetic recovery as an alternative to Biot-Savart `Hz`, and add a dedicated dBzdt receiver-recovery diagnostic.
4. Continue P3 by wiring `PronyConductivity` into DOLFINx total-field IP assembly and adding solver-level `delta_sigma=0` no-IP equivalence tests.
5. Continue P4 by implementing real `EmpymodPrimaryProvider` sampling or a 1D reference backend.
6. Continue P5 by adding the DOLFINx scalar DC secondary solve for `phi_s`.
7. Continue P6 by wiring the step kernels to DOLFINx FEM operators and receiver operators.
8. Continue P7 by connecting `validation_3comp` to real no-IP/IP empymod or 1D reference runs over `1e-5 s <= t_obs <= 1 s`.
9. Continue P8 by generating a gmsh terrain/leakage mesh and running a small DOLFINx forward example.
