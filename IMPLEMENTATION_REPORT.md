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
- Average receiver sampling and simultaneous point/average diagnostic CSV/PNG
  artifact output are implemented and smoke-tested for the E-form DOLFINx
  verification pipeline. H-form diagnostic output still writes only the main
  receiver response.
- Faraday-integrated `Hz` recovery is not implemented in this round.
- P3 currently provides the material API and memory-update tests; DOLFINx total-field IP assembly still needs to be migrated to this API and verified against no-IP when `delta_sigma=0`.
- P4 currently provides zero/cached primary providers and receiver-side empymod primary sampling through an injected/reference runner; FEM-space primary field interpolation remains pending.
- P5 currently provides a pure initialization core with an injected secondary field solver; DOLFINx scalar Poisson assembly for `phi_s` remains pending.
- P6 currently provides pure no-IP/IP time-step kernels with injected secondary solvers; DOLFINx curl-curl/mass/Robin operator assembly remains pending.
- P7 currently verifies artifact generation from supplied arrays; it does not yet run a real empymod/1D backend to prove 5% physical agreement.
- P7 CLI currently reads precomputed prediction/reference CSV files; it does not yet launch DOLFINx or empymod itself.
- `atem3d-validate-empymod --artifact-dir` bridges real validation results to artifact files, but final 5% agreement still depends on the underlying simulation/reference result.
- P8 currently verifies marker/material/channel geometry utilities; it does not yet run a DOLFINx gmsh complex-terrain forward example.
- Full no-IP/IP 5% acceptance is not yet achieved.
- `compute_error` and the validation artifact writer now share the task-book
  floor policy. Older reports generated before this change should be
  regenerated with `--postprocess-partial` before comparing error numbers.

## Next Steps

1. Add real FEM source residual diagnostics using assembled gradient/divergence/curl operators.
2. Extend the latest-model point/disk diagnostic run beyond the first five output times after improving the receiver/curl recovery path, so long runs are not spent confirming the same early-time failure.
3. Add Faraday-integrated magnetic recovery as an alternative to Biot-Savart `Hz`, and add a dedicated dBzdt receiver-recovery diagnostic.
4. Continue P3 by wiring `PronyConductivity` into DOLFINx total-field IP assembly and adding solver-level `delta_sigma=0` no-IP equivalence tests.
5. Continue P4 by implementing real `EmpymodPrimaryProvider` sampling or a 1D reference backend.
6. Continue P5 by adding the DOLFINx scalar DC secondary solve for `phi_s`.
7. Continue P6 by wiring the step kernels to DOLFINx FEM operators and receiver operators.
8. Continue P7 by connecting `validation_3comp` to real no-IP/IP empymod or 1D reference runs over `1e-5 s <= t_obs <= 1 s`.
9. Continue P8 by generating a gmsh terrain/leakage mesh and running a small DOLFINx forward example.
