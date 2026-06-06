# P0-P8 Implementation Report

## Scope

This report covers the implementation rounds currently committed or staged from the task book:

- P0: freeze the current total-field baseline.
- P1: add waveform interfaces, full turn-off time grid, and interval-average `dI/dt`.
- P2: add no-IP three-component validation artifacts and diagnostics.
- P3 partial: add a Debye/Prony material conductivity API and pure-Python memory tests.
- P4 partial: add primary-field provider interfaces with zero/cached providers and runner-backed empymod sampling.
- P5 partial: add a DC secondary initialization core with zero-contrast and
  nonzero-contrast smoke tests.
- P6 partial: add no-IP/IP primary-secondary TDEM step kernels with zero-contrast tests.
- P7 partial: add no-IP/IP three-component validation smoke artifact writer.
- P8 partial: add complex-terrain leakage-channel material-map utilities,
  a small DOLFINx primary-secondary forward smoke, and a generated Gmsh
  terrain/leakage forward smoke.
- P7 CLI: add `validate-noip-3comp` and `validate-ip-3comp` artifact commands from CSV inputs.
- P7 acceptance gate: add a combined no-IP/IP final acceptance report command
  that reads both `error_summary.json` files and fails unless both cases have
  `final_acceptance_passed=true`.
- Corrected-model spec helper: add a canonical source/receiver/time-window
  configuration for the latest task geometry and a CLI to write no-IP/IP case
  specs, including runner/material metadata.
- Corrected-model runner scaffold: add a pure Python validation runner and
  `tdem-ip-forward corrected-model-run` CLI that can write full artifact sets
  from injected forward/reference runners. The default DOLFINx forward backend
  now has a memory-safe no-IP zero-secondary smoke path; nonzero-contrast/IP
  corrected-model backend validation remains pending.
- Corrected-model schematic integration: `corrected-model-run` now writes
  `model_schematic.png` into each case output directory and records schematic
  metadata in `diagnostics.json`.
- Corrected-model polarization-effect integration: `corrected-model-run --case
  both` now writes a sibling `polarization_effect/` directory with IP-minus-noIP
  response, reference, error, summary, and plot artifacts.

It does not claim that the full 1e-5 s to 1 s 5% accuracy target is achieved.
Validation artifacts now write an explicit `acceptance_status` object and a
top-level `final_acceptance_passed` flag. A run can only set
`final_acceptance_passed=true` when it is explicitly marked
`validation_scope=corrected_model_full`, covers `1e-5 s` to `1 s`, includes
`Ex`, `Ey`, and `Hz` or `dBzdt`, uses an allowed empymod/1D reference, keeps
the threshold at or below `5%`, and passes the physical error gate. Smoke or
partial-window runs remain useful diagnostics but are blocked from claiming
final acceptance.

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

- `src/atem3d/final_acceptance.py`
  - `summarize_final_acceptance`
  - `write_final_acceptance_report`
  - Writes `final_acceptance_summary.json` and
    `final_acceptance_report.txt` from no-IP/IP `error_summary.json` files.
  - Requires both no-IP and IP cases to pass their own
    `final_acceptance_passed` gates before the combined final gate passes.
  - Optionally reads no-IP/IP `diagnostics.json` files and propagates
    `validation_failure.reason_codes` plus structured diagnostic check status
    into `failure_diagnostics_by_case` in the combined final report.

- `src/atem3d/yaml_io.py`
  - `safe_dump_yaml`
  - `safe_load_yaml`
  - Uses `PyYAML` when available and falls back to small built-in YAML
    writer/reader helpers for dictionaries/lists/scalars when the WSL FEniCSx
    environment does not provide `yaml`.

- `src/atem3d/corrected_model.py`
  - `CorrectedModelValidationConfig`
  - `build_corrected_model_case_specs`
  - `build_corrected_leakage_channel_case_specs`
  - `build_published_paper_model_target_spec`
  - Stores the corrected latest geometry:
    source `(-500, 200, -0.1) -> (500, 200, -0.1)`, receiver
    `(0, -300, -0.1)`, current `10 A`, source length `1000 m`, parallel
    offset `500 m`.
  - Generates full-window log observation times from `1e-5 s` to `1 s` and
    no-IP/IP case metadata with `validation_scope=corrected_model_full`.
  - Generates a memory-safe corrected-scale leakage-channel diagnostic spec
    with the same source/receiver geometry, a `4000 m x 4000 m x 1100 m` box,
    coarse `2 x 2 x 1` DOLFINx cells, and no-IP/IP leakage-channel material
    metadata.
  - Records a published-paper reproduction target for the Journal of Applied
    Geophysics paper `Analysis of 3D induced polarization effects of SOTEM`
    (`S092698512400329X`) without treating missing full-text model parameters
    as verified.

- `src/atem3d/model_schematic.py`
  - `write_model_schematic`
  - Writes a two-panel plan/depth PNG schematic from a corrected-model case
    spec, including source line, receiver, computational domain, and optional
    leakage-channel polyline.

- `src/atem3d/polarization_effect.py`
  - `write_polarization_effect_artifacts`
  - Reads no-IP/IP validation artifact directories and writes IP-minus-noIP
    polarization-effect response CSV, reference CSV, error CSV, summary JSON,
    response comparison PNG, and error-curve PNG.

- `src/atem3d/source_diagnostics.py`
  - `diagnose_source_consistency`
  - Reports task-book source residuals:
    `source_endpoint_balance_residual`,
    `dc_current_conservation_residual`,
    `initial_curl_residual`, and
    `waveform_integral_residual`.

- `dolfinx/sotem_pipeline.py`
  - Default source/receiver geometry now matches the corrected latest model:
    source `(-500, 200, -0.1) -> (500, 200, -0.1)`, receiver
    `(0, -300, -0.1)`, source current `10 A`, expected source length
    `1000 m`, and expected parallel offset `500 m`.
  - Added `min_steps_during_turnoff`.
  - Added `min_steps_before_first_observation` for after-ramp internal
    substeps between ramp-off end and the first observation output.
  - Added `divergence_cleaning_strength` to scale the conductivity
    divergence-cleaning correction for diagnostics.
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
  - `diagnose_source_consistency` can now optionally delegate explicit
    matrix/weak-form diagnostic inputs to the reusable
    `atem3d.source_diagnostics` core while preserving the old self-contained
    fallback path.
  - E-form runs now measure the residual initial-field curl,
    `-curl(E_initial)`, and write it to both `diagnostics["initial_field"]`
    and `diagnostics["source_consistency"]["initial_curl_residual"]`.
  - `write_validation_artifacts` and `write_source_only_diagnostics` now read
    `source_info["consistency_diagnostic_inputs"]` and write those computed
    residuals into `diagnostics.json`.
  - Receiver diagnostics now record sampling/cell-candidate statistics:
    `sample_count`, `candidate_count_min`, `candidate_count_max`, and
    `candidate_count_mean` in `receiver_diagnostics.csv`.
  - Added optional receiver magnetic-rate mode:
    - `--magnetic-dbdt-mode curl` keeps the original E-form `-curl(E)` output.
    - `--magnetic-dbdt-mode biot_rate` uses the finite-difference rate of the
      Biot-Savart receiver `H` as a diagnostic `dBzdt` path.
  - Added optional receiver magnetic-field recovery mode:
    - `--magnetic-receiver-mode faraday_integrated` initializes receiver `Hz`
      from the existing total-current Biot-Savart value and advances it with
      backward-Euler `B^{n+1}=B^n+dt*dBzdt(E^{n+1})`.
    - This mode exposes a direct Faraday-integrated `Hz` diagnostic path for
      comparison against Biot-Savart recovery; it is not yet a full-window
      acceptance result.
  - Added `_make_secondary_receiver_projector_from_evaluate_receivers`, a
    DOLFINx-side bridge that reuses existing `evaluate_receivers(E, dBdt, ...)`
    sampling to produce the component table expected by the reusable
    primary-secondary forward core. This is a receiver bridge only; full
    DOLFINx primary-secondary operator assembly remains pending.
  - Added `_make_dolfinx_zero_secondary_receiver_projector` and a WSL
    primary-secondary zero-contrast forward smoke that wires
    `PrimarySecondaryForwardOperator`, `CachedPrimaryProvider`, and DOLFINx
    receiver sampling together and verifies total receiver response equals the
    primary response.
  - Added `_solve_dc_secondary_field`, a DOLFINx scalar secondary-potential
    initializer for the primary-secondary DC problem. The zero-contrast path
    is WSL-tested to return near-zero `Es0`; a nonzero-contrast unit-cube
    smoke test verifies finite nonzero `Es0` and a converged KSP solve. Full
    corrected-model primary-secondary integration remains pending.
  - Added `_make_dolfinx_secondary_step_solver`, a DOLFINx-backed
    primary-secondary TDEM step solver hook that forms
    `K + M_sigma/dt (+ Robin)` and solves with the existing AMS setup. The
    zero-RHS path is WSL-tested to return zero secondary samples; a nonzero
    constant sample RHS hook is WSL-tested through PETSc solve and
    `solution_to_samples`.
  - Added `_make_nedelec_rhs_interpolator_from_samples` for converting
    secondary RHS sample tables into Nedelec Functions. Constant sample tables
    are supported directly with a scale-aware tolerance for roundoff-level
    sample noise; non-constant tables require explicit sample coordinates.
  - Added `_nedelec_interpolation_points` for exporting physical Nedelec
    interpolation points from local tetrahedral cells. A WSL smoke verifies
    these points can drive non-constant tabulated RHS interpolation and
    sampling, which is the coordinate bridge needed by production primary
    providers.
  - Added `_make_dolfinx_primary_secondary_forward_operator`, which exports
    the physical Nedelec interpolation points, samples a `PrimaryFieldProvider`
    on those points through `PrimarySecondaryForwardOperator`, and wires the
    DOLFINx secondary initializer, stepper, and receiver projector. A WSL
    smoke verifies the provider receives the actual Nedelec point table.
  - Added `_make_dolfinx_materials_from_cell_material_map`, which converts a
    `CellMaterialMap` into DG0 `sigma`, `sigma_initial`, `sigma_infinity`,
    `mu_inv`, representative Prony material metadata, and optional spatial
    Debye `delta_functions`. A WSL leakage-channel smoke uses this path in a
    small DOLFINx primary-secondary forward run.
  - Added `_write_small_gmsh_terrain_leakage_mesh`, which writes a tiny
    Gmsh terrain volume with physical earth/outer/top-surface tags. A WSL
    smoke loads the generated mesh, marks a leakage channel from cell centers,
    maps it to DG0 IP materials, and runs one primary-secondary forward step.
  - Added `_make_nedelec_solution_sampler_at_points` for sampling DOLFINx
    Nedelec Functions back to `(n_samples, 3)` tables. The secondary step smoke
    now exercises `rhs_to_function` plus this production `solution_to_samples`
    path with nonzero RHS.
  - Added an optional `secondary_state_stepper` hook to
    `PrimarySecondaryForwardOperator` for FEM backends that must advance
    state in native Function form rather than scalar sample tables.
  - Added an optional `secondary_state_initializer` hook so FEM backends can
    initialize DC secondary state directly from `Ep0` samples without reducing
    spatially varying material contrast to a scalar callback.
  - Added `_make_dolfinx_primary_secondary_forward_adapters`, a DOLFINx adapter
    that wires the scalar DC secondary solve, secondary step solver, Nedelec
    sample/function conversion, solution sampler, and receiver projector into
    `PrimarySecondaryForwardOperator`. The no-IP transient path now keeps
    variable DG0 material current contrast and RHS in DOLFINx Function form.
    WSL unit-cube forward smokes verify both uniform and variable nonzero
    contrast paths produce finite nonzero secondary contributions. A scalar
    Debye/Prony IP state-stepper smoke now updates Function-level `chi_k`,
    `c_new`, and `deltaJ` with the shared `PronyConductivity` coefficients;
    a spatial DG0 `delta_sigma` smoke now uses `debye["delta_functions"]` for
    Function-level IP current contrast.
  - Validation summaries now distinguish strict component-wise acceptance from
    physical acceptance:
    - `pass_all_components` remains the strict per-component robust-error gate.
    - `physical_pass_all_components` allows weak horizontal electric components
      such as near-zero `Ey` to pass by absolute error scaled by
      `max(|Eh_ref|)`.
    - `weak_components`, `weak_component_passed`, and
      `physical_failed_components` are written to `error_summary.json`.
    - `acceptance_status` and `final_acceptance_passed` now guard final
      task-book acceptance separately from smoke/partial diagnostic runs.
      `diagnostics.json` also records acceptance blocking reasons.
  - Biot-Savart receiver `Hz` and optional `biot_rate` now use the same
    point/average receiver sampling points as `Ex/Ey` and curl `dBzdt`.

- `dolfinx/legacy_total_field_baseline.py`
  - Frozen copy of the current total-field baseline implementation.

- `dolfinx/sotem_pipeline.py`
  - `_interpolate_vector_callable_to_nedelec_function` centralizes interpolation
    of vector callables into the Nedelec `V` space.
  - The analytic halfspace DC initializer now uses this shared helper.

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
    `EmpymodSurvey` import now also avoids import-time `yaml` dependency
    unless HDF5 config parsing is actually used.
  - `EmpymodPrimaryProvider` now defaults to
    `atem3d.empymod_compare.run_empymod_reference` for transient receiver and
    FEM-point `Ex/Ey/Ez` or `dB/dt` sampling when no injected
    `reference_runner` is supplied. Construction still avoids importing
    empymod.
  - `EmpymodPrimaryProvider.get_Ep_dc_on_V` now defaults to the
    uniform-halfspace analytic grounded-wire DC backend when no injected
    `dc_runner` is supplied. Layered or complex-background DC primary fields
    still need an explicit `dc_runner`.
  - Injected-runner receiver `E` and `dBdt` primary sampling via `EmpymodSurvey`.
  - Injected-runner FEM point primary `E_p(t)` sampling through `get_Ep_on_V`.
  - `PrimaryFEMInterpolator` for provider-to-FEM-point `E_p(t)` and `E_p,dc`
    sampling, with an injected assembler hook for later DOLFINx Function/vector
    interpolation.

- `src/atem3d/forward_operator.py`
  - `ForwardRequest`
  - `ForwardOperator.forward(model, survey, waveform, times)`
  - Injected runner backend and strict time/data-shape validation for
    inversion-ready forward calls.
  - `TabulatedVectorField` and `make_tabulated_vector_assembler` for converting
    fixed point/vector samples into a DOLFINx `Function.interpolate`-style
    callable with `(3, n)` coordinate input and `(3, n)` component output.

- `src/atem3d/solvers/primary_secondary_forward.py`
  - `PrimarySecondaryForwardOperator`
  - Pure primary-secondary forward orchestration around
    `PrimaryFieldProvider`, `PrimaryFEMInterpolator`, DC secondary
    initialization, and no-IP/IP secondary time-step kernels.
  - Keeps DOLFINx-specific pieces injectable:
    `secondary_field_solver`, `secondary_step_solver`, and
    `secondary_receiver_projector`.
  - `secondary_receiver_projector` can return either a flattened receiver row
    or a natural `(n_receivers, n_components)` table, which is then flattened
    into the validation/output ordering.
  - Zero-contrast no-IP response returns the primary receiver `Ex`, `Ey`, and
    `dBdt` components without requiring a secondary solve.

- `src/atem3d/solvers/receiver_projection.py`
  - `SecondaryReceiverProjection`
  - Pure adapter from injected secondary electric-field and `dB/dt` receiver
    samplers to ordered component tables such as `Ex`, `Ey`, and `dBzdt`.
  - Intended DOLFINx hook point for projecting secondary FEM fields into the
    primary-secondary forward core without embedding DOLFINx imports in the
    reusable solver package.

- `src/atem3d/receivers.py`
  - `PointReceiver`
  - `AverageReceiver` for deterministic `disk_average` and `volume_average`
    sampling around a point receiver.
  - `AverageReceiver` supports recovered magnetic-field vector sampling with
    the same `H`/`B` component convention as `PointReceiver`.
  - `build_receiver` factory for `point`, `disk_average`, and
    `volume_average` receiver configuration.

- `src/atem3d/__init__.py`
  - Exposes pure receiver APIs: `PointReceiver`, `AverageReceiver`, and
    `build_receiver` without requiring optional modelling dependencies.

- `src/atem3d/config.py`
  - Receiver YAML entries can now pass `type`/`receiver_type` and `radius`
    through `build_receiver`.

- `src/atem3d/solvers/dc_secondary.py`
  - `DCSecondaryInitialization`
  - `initialize_dc_secondary`
  - `initialize_dc_secondary_from_primary`
  - Zero-contrast exact `Es0=0` path.
  - IP memory initialization `chi0 = Ep0 + Es0`.
  - Initial secondary current contrast `deltaJ0`.

- `src/atem3d/solvers/tdem_secondary.py`
  - `SecondaryState`
  - `secondary_state_from_dc_initialization`
  - `secondary_step_noip`
  - `secondary_step_ip`
  - Variable-`dt` backward-Euler secondary RHS density helpers through injected solvers.
  - Exact zero-contrast secondary response for no-IP and zero-delta IP.

- `src/atem3d/validation_3comp.py`
  - `ThreeComponentValidationInput`
  - `write_three_component_validation_artifacts`
  - Enforces the task-book validation window coverage:
    `1e-5 s <= t_obs <= 1 s`.
  - Required no-IP/IP validation CSV/JSON/PNG output names.
  - `run_config_resolved.yaml` resolved validation configuration output, now
    written as YAML rather than JSON text with a YAML extension.
  - Robust relative error and peak-normalized error output through existing metric code.
  - IP Prony metadata in `error_summary.json`.
  - `diagnostics.json` now includes `validation_failure` with failed
    components/times, reason codes, and the task-book diagnostic check order
    whenever a validation table is written.
  - Failed validations now also write structured diagnostic checks for
    `time_step_error`, `mesh_error`, `boundary_error`, `source_term_error`,
    `receiver_sampling_error`, `magnetic_recovery_error`, and
    `ip_memory_error`. Each check has a status, evidence dictionary, and
    recommended action so runs that exceed `5%` point directly to the required
    follow-up diagnostics instead of only reporting `failed`.

- `src/atem3d/cli.py`
  - Keeps legacy simulation run mode.
  - Adds task-book `run CONFIG.yaml` subcommand dispatch while preserving
    legacy `CONFIG.yaml` invocation.
  - Adds `validate-noip-3comp CONFIG.yaml`.
  - Adds `validate-ip-3comp CONFIG.yaml`.
  - Adds `validate-secondary CONFIG.yaml` for a lightweight primary-secondary
    zero-contrast validation summary, per-time trace CSV, diagnostics JSON,
    and resolved YAML config.
  - `validate-secondary` can now use `PrimarySecondaryForwardOperator` when
    receiver locations, components, and primary receiver `E`/`dBdt` tables are
    supplied, writing `primary_secondary_predictions.csv` and checking total
    receiver response against the primary response in the zero-contrast case.
  - `validate-secondary` reads optional Prony material metadata and uses the
    IP secondary step path when Debye terms are present; zero-delta IP material
    is tested to remain equivalent to the no-IP zero-contrast response.
  - Adds `plot RUN_DIR` to regenerate `comparison_3comp.png` and
    `error_curves_3comp.png` from validation CSV artifacts.
  - Adds `model-schematic SPEC.json --case noip|ip --output FIG.png` for
    writing model geometry schematics from corrected-model specs.
  - Adds `polarization-effect NOIP_DIR IP_DIR --output-dir OUT_DIR` for
    writing dedicated IP-minus-noIP response/error artifacts.
  - Adds `corrected-leakage-model-spec` for writing a corrected-scale,
    memory-safe leakage-channel diagnostic spec before running DOLFINx.
  - Adds `published-paper-model-spec` for writing the published SOTEM paper
    reproduction target metadata and the list of full-text parameters still
    required before a paper-response overlay can be claimed.
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

- `tests/test_forward_operator.py`
- `tests/test_waveform.py`
- `tests/test_time_grid.py`
- `tests/test_error_metric_floor.py`
- `tests/test_dolfinx_validation_artifacts.py`
- `tests/test_dolfinx_biot_receiver.py`
- `tests/test_dolfinx_dc_secondary_solver.py`
- `tests/test_dolfinx_primary_secondary_forward_smoke.py`
- `tests/test_dolfinx_secondary_step_solver.py`
- `tests/test_prony.py`
- `tests/test_primary_provider.py`
- `tests/test_dc_initialization.py`
- `tests/test_secondary_zero_contrast.py`
- `tests/test_primary_secondary_forward.py`
- `tests/test_secondary_receiver_projection.py`
- `tests/test_noip_3comp_validation_smoke.py`
- `tests/test_ip_3comp_validation_smoke.py`
- `tests/test_complex_terrain_leakage_smoke.py`
- `tests/test_dolfinx_complex_terrain_leakage_forward.py`
- `tests/test_validation_3comp_cli.py`
- `tests/test_source_consistency.py`
- `tests/test_average_receivers.py`
- `tests/test_public_api.py`
- `tests/test_model_schematic.py`
- `tests/test_polarization_effect.py`
- Existing `tests/test_empymod_validation.py`
- Existing `tests/test_empymod_validation_cli.py`

## Validation Command

WSL/DOLFINx environment check:

```bash
python dolfinx/sotem_pipeline.py --check-env-only --no-install
```

Latest observed in WSL `Ubuntu` / `fenicsx` conda environment on 2026-06-06:

- DOLFINx `0.8.0`
- empymod `2.5.4`
- gmsh `4.15.2`
- meshio `5.3.5`
- PETSc hypre support: `True`
- pipeline reports regularized volume source fallback unless explicitly overridden.
- WSL was shut down after the check and confirmed `Stopped` with `wsl -l -v`.

Full Windows/Python test audit:

```bash
python -m pip install --user --no-cache-dir -e .
python -m pytest -q
```

Observed result after installing the declared project dependencies:

```text
all collected tests passed
2 skipped
```

The two skipped tests are optional Pardiso solver tests in
`tests/test_solver_core.py`, skipped because `pydiso` is not installed in the
Windows Python environment. The first install attempt without `--no-cache-dir`
failed on a local pip cache permission error for `pymatsolver`; rerunning with
`--user --no-cache-dir` installed the declared dependencies successfully.

Latest WSL/DOLFINx primary-secondary and leakage smoke audit:

```bash
PYTHONPATH=src /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest -q tests/test_dolfinx_primary_secondary_forward_smoke.py tests/test_dolfinx_complex_terrain_leakage_forward.py
```

Observed result:

```text
13 passed
```

This WSL test set covers DOLFINx primary-secondary forward adapters, corrected
runner no-IP/IP smoke paths, leakage-channel material mapping, and generated
Gmsh terrain/leakage forward smoke. WSL was shut down after the run and
confirmed `Ubuntu Stopped`.

Latest WSL source-only smoke for the corrected explicit geometry:

```text
workdir = dolfinx/current_task_runs/y200_rxminus300_source_only_latest_smoke
source = (-500, 200, -0.1) -> (500, 200, -0.1)
receiver = (0, -300, -0.1)
expected_source_length = 1000 m
expected_parallel_offset = 500 m
mode = manual_line
```

Result:

```text
source_endpoint_balance_residual = 2.2680708322072504e-14
waveform_integral_residual = 0.0
quadrature_points = 1200
missed_points = 0
unique_hit_cells = 200
runtime_total_seconds = 4.05504921
```

Interpretation: the latest explicit source geometry now passes the source-only
endpoint-balance and waveform-integral diagnostics on a coarse smoke mesh. This
does not validate the transient PDE response or the final `5%` no-IP/IP target.
WSL was shut down after the run and confirmed `Stopped`.

Latest WSL no-IP transient first-output smoke for the corrected explicit geometry:

```text
workdir = dolfinx/current_task_runs/y200_rxminus300_noip_first_output_latest_smoke
source = (-500, 200, -0.1) -> (500, 200, -0.1)
receiver = (0, -300, -0.1)
expected_source_length = 1000 m
expected_parallel_offset = 500 m
t_obs = 1.0e-5 s
stop_after_outputs = 1
```

Runtime:

```text
total = 16.123 s
mesh = 1.760 s
setup = 1.032 s
forward_solve = 9.651 s
empymod_reference = 0.591 s
```

Result at the first output:

```text
max_error_Ex = 0.5726338805279592
max_error_Ey = 4134147469.564552
max_error_dBzdt = 0.3370050078629801
physical_failed_components = Ex, dBzdt
weak_components = Ey
weak_component_passed = true
source_endpoint_balance_residual = 2.2680708322072504e-14
waveform_integral_residual = 0.0
receiver_sampling_issue_suspected = true
disk_average_improves_Ey_only = true
```

Interpretation: the transient path runs and writes validation artifacts for the
latest explicit geometry, but the first output still fails the `5%` physical
gate for `Ex` and `dBzdt`. The source balance and waveform checks pass, so the
next diagnostic focus should remain mesh/boundary/receiver/curl recovery and
the primary-secondary formulation. WSL was shut down after the run and
confirmed `Stopped`.

First-output local mesh sensitivity smoke:

```text
workdir = dolfinx/current_task_runs/y200_rxminus300_noip_first_output_mesh100_smoke
source_mesh_size = 100 m
receiver_mesh_size = 100 m
t_obs = 1.0e-5 s
stop_after_outputs = 1
```

Runtime:

```text
total = 19.615 s
mesh = 2.006 s
setup = 1.297 s
forward_solve = 12.252 s
empymod_reference = 0.633 s
```

Result:

```text
max_error_Ex = 0.7429212596469872
max_error_Ey = 868019039.4803351
max_error_dBzdt = 0.5860474600297446
physical_failed_components = Ex, dBzdt
weak_components = Ey
weak_component_passed = true
source_endpoint_balance_residual = 2.6473569601796998e-14
waveform_integral_residual = 0.0
quadrature_points = 2200
missed_points = 0
```

Compared with the `source_mesh_size=receiver_mesh_size=200 m` first-output
smoke, simply lowering both mesh sizes to `100 m` did not improve the physical
gate:

```text
Ex:    0.5726338805279592 -> 0.7429212596469872
dBzdt: 0.3370050078629801 -> 0.5860474600297446
```

Interpretation: the first-output error is not a simple monotonic local mesh
size issue on this coarse geometry. The receiver/cell-candidate geometry,
source transfer, boundary/domain setup, and primary-secondary formulation are
still higher-value next diagnostics than a blind local mesh-size sweep. WSL was
shut down after the run and confirmed `Stopped`.

Receiver cell-candidate geometry diagnostic update:

- `dolfinx/sotem_pipeline.py` now writes additional receiver diagnostic fields
  for candidate-cell geometry:
  `multi_candidate_sample_count`, candidate-center distance min/max/mean,
  selected-center distance mean/max, candidate-center z min/max, and
  selected-center z mean.
- These fields are written to `receiver_diagnostics.csv` and persisted through
  forward checkpoint/partial payloads.
- The default lightweight receiver path remains unchanged unless receiver
  diagnostics are requested or the evaluation mode is `nearest_center` or
  `shallowest`.

WSL coarse receiver-geometry smoke for the corrected explicit geometry:

```text
workdir = dolfinx/current_task_runs/y200_rxminus300_receiver_geometry_diag_coarse
source_mesh_size = 200 m
receiver_mesh_size = 200 m
receiver_evaluation_mode = nearest_center
receiver_diagnostic_types = point,disk_average,volume_average
t_obs = 1.0e-5 s
stop_after_outputs = 1
```

Runtime outcome:

```text
mesh nodes = 2434
mesh cells reported by DOLFINx = 12575
Nedelec dofs = 15330
estimated solver memory = 0.4299255 GB
WSL wall time = 40.2 s
WSL shutdown state after run = Ubuntu Stopped
```

First-output error summary:

```text
point Ex robust error = 0.4934159805166578
point Ey robust error = 21390432199.75224
point Hz robust error = 0.18925507396055294
point dBzdt robust error = 0.011191680824714136
```

Receiver-geometry diagnostic at the point receiver:

```text
sample_count = 1
candidate_count = 18
candidate_center_distance_min = 30.683430835714788 m
candidate_center_distance_max = 95.46041877884072 m
candidate_center_distance_mean = 54.74736420492263 m
candidate_center_z_min = -89.49570752684954 m
candidate_center_z_max = -0.025 m
selected_center_distance_mean = 30.683430835714788 m
selected_center_z_mean = -0.025 m
```

Interpretation: the coarse receiver cell geometry is not a reliable point
evaluation environment. The selected near-surface cell is still about `30.7 m`
from the receiver, while colliding candidates extend down to about `89.5 m`.
The point `dBzdt` happened to pass the 5% gate at the first output, but `Ex`,
`Hz`, and the weak-component-scaled `Ey` did not. Disk/volume averaging did not
improve `Ex` or `dBzdt` against empymod in this smoke. This supports continuing
with receiver/source-aligned mesh construction and the primary-secondary
formulation rather than treating averaging as a fix.

Receiver anchor mesh-size control:

- `PipelineConfig.receiver_anchor_mesh_size` and CLI option
  `--receiver-anchor-mesh-size` were added. A value of `0` preserves the
  existing behavior and reuses `receiver_mesh_size`.
- When positive, this anchor size controls the receiver embedded point, receiver
  cloud, receiver surface cloud, receiver distance field `SizeMin`, receiver
  ball field `VIn`, and global `MeshSizeMin` lower bound.
- This allows local receiver anchoring to be tightened without forcing the
  whole receiver refinement radius or source mesh to use the same small size.

WSL coarse receiver-anchor smoke for the corrected explicit geometry:

```text
workdir = dolfinx/current_task_runs/y200_rxminus300_receiver_anchor10_diag_smoke
source_mesh_size = 200 m
receiver_mesh_size = 200 m
receiver_anchor_mesh_size = 10 m
receiver_refinement_radius = 60 m
receiver_evaluation_mode = nearest_center
receiver_diagnostic_types = point,disk_average,volume_average
t_obs = 1.0e-5 s
stop_after_outputs = 1
```

Runtime outcome:

```text
mesh nodes = 5354
mesh cells reported by DOLFINx = 30353
Nedelec dofs = 36028
estimated solver memory = 0.9755205 GB
WSL wall time = 34.5 s
WSL shutdown state after run = Ubuntu Stopped
```

Point receiver geometry improved relative to the previous coarse smoke:

```text
candidate_count = 32
candidate_center_distance_min = 3.9535585236594133 m
candidate_center_distance_max = 6.837442869962425 m
candidate_center_distance_mean = 5.879788398175545 m
candidate_center_z_min = -5.074999999999999 m
candidate_center_z_max = -0.025 m
selected_center_distance_mean = 3.9535585236594133 m
selected_center_z_mean = -0.025 m
```

First-output error summary:

```text
point Ex robust error = 0.42293716574207946
point Ey robust error = 7661803478.60315
point dBzdt robust error = 0.3715145540364427
```

Interpretation: the anchor option substantially improves local receiver
geometry and reduces `Ex` and weak-component-scaled `Ey` error compared with
the previous coarse diagnostic, but it worsens `dBzdt` at this first output.
Receiver geometry is therefore confirmed as one error source, but not the full
root cause. The next high-value checks are curl/dBdt recovery consistency,
source transfer/source-channel alignment, and the primary-secondary solver path.

Magnetic-component validation summary update:

- `dolfinx/sotem_pipeline.py` and `atem3d.metrics.robust_component_errors`
  now keep per-component magnetic summary fields when both `Hz` and `dBzdt`
  are present:
  `max_error_Hz`, `rms_error_Hz`, `max_peak_normalized_error_Hz`,
  `max_error_dBzdt`, `rms_error_dBzdt`, and
  `max_peak_normalized_error_dBzdt`.
- The legacy `Hz_or_dBzdt` fields remain for compatibility and still track the
  active/last magnetic validation quantity.
- `error_summary.json` now includes `magnetic_components`, e.g.
  `["Hz", "dBzdt"]`, so later WSL smoke runs can distinguish magnetic-field
  recovery failure from `-curl(E)` magnetic-rate failure.

Validation command for this update:

```bash
python -m pytest -q tests/test_dolfinx_validation_artifacts.py tests/test_error_metric_floor.py tests/test_noip_3comp_validation_smoke.py tests/test_ip_3comp_validation_smoke.py tests/test_validation_3comp_cli.py
```

Observed result:

```text
28 passed
WSL state after checks = Ubuntu Stopped
```

Lightweight P0-P2 tests:

```bash
python -m pytest -q tests/test_waveform.py tests/test_time_grid.py tests/test_source_consistency.py tests/test_average_receivers.py tests/test_error_metric_floor.py tests/test_dolfinx_validation_artifacts.py tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_solves_through_ramp_then_returns_observation_times tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_keeps_ramp_grid_when_observations_end_before_ramp tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_uses_ramp_solver_t_min_before_later_observation_start
```

P3 material-interface tests:

```bash
python -m pytest -q tests/test_prony.py tests/test_ip_model.py tests/test_debye_fit.py
```

P4 primary-provider tests:

```bash
python -m pytest -q tests/test_primary_provider.py
```

Current P4 status:

- `EmpymodPrimaryProvider.get_receiver_E` and `get_receiver_dBdt` build an
  `EmpymodSurvey` and can use either an injected `reference_runner` or the
  default `run_empymod_reference` backend. Tests still use injection/monkeypatch
  where needed, so most tests do not depend on importing or running empymod.
- `EmpymodPrimaryProvider.get_Ep_on_V` now uses the same injected runner to
  sample `Ex/Ey/Ez` at arbitrary FEM point coordinates. This is the first
  runner-backed FEM-space primary field sampling hook needed by the
  primary-secondary solver path.
- `EmpymodPrimaryProvider.get_Ep_dc_on_V` now accepts a dedicated injected
  `dc_runner(points, config=..., **dc_kwargs)` and validates that it returns an
  `(n_points, 3)` DC primary field table. If no `dc_runner` is injected, it
  falls back to `analytic_halfspace_dc_runner` for the uniform-halfspace
  grounded-wire DC primary.
- A local default-runner smoke using the corrected model, `srcpts=5`, receiver
  `(0, -300, -0.1)`, and `t_obs=1e-5 s` returned finite values:
  `E=[-2.27169895e-04, 3.58434271e-19, 0.0]` and
  `dBdt=[0.0, 0.0, -4.34444925e-18]`.
- `analytic_halfspace_grounded_wire_dc_electric_field` and
  `analytic_halfspace_dc_runner` provide a concrete uniform-halfspace grounded
  wire DC primary backend matching the DOLFINx analytic-DC formula.
- `PrimaryFEMInterpolator` samples `E_p(t)` and `E_p,dc` on fixed FEM point
  coordinates, validates `(n_points, 3)` shapes, supports multi-time sampling,
  and optionally passes the point/vector table to an injected assembler. This is
  still a pure adapter.
- `TabulatedVectorField` provides the first concrete DOLFINx-compatible callable
  adapter for tabulated primary samples. It maps queried interpolation points
  back to the sampled table and returns component-major values suitable for
  `Function.interpolate`.
- `dolfinx/sotem_pipeline.py` now has a shared
  `_interpolate_vector_callable_to_nedelec_function` helper for inserting vector
  callables into the Nedelec function space. The analytic halfspace DC initial
  field uses this helper, giving the primary-secondary path a concrete DOLFINx
  insertion point to reuse.

P5 DC secondary initialization tests:

```bash
python -m pytest -q tests/test_dc_initialization.py
```

Current P5 status:

- `initialize_dc_secondary_from_primary` now pulls `E_p,dc` from a
  `PrimaryFEMInterpolator`-compatible object via `sample_Ep_dc()` and delegates
  to the existing DC secondary initialization core.
- This creates a tested bridge from P4 primary sampling into P5 DC secondary
  state construction. The actual DOLFINx scalar Poisson solve for `phi_s` is
  still injected rather than assembled in the pipeline.

P6 TDEM secondary stepper tests:

```bash
python -m pytest -q tests/test_secondary_zero_contrast.py
```

P6 secondary validation CLI regression:

```bash
python -m pytest -q tests/test_cli_subcommands.py::test_cli_validate_secondary_writes_zero_contrast_summary
```

`tdem-ip-forward validate-secondary CONFIG.yaml` now writes:

```text
secondary_validation_summary.json
secondary_validation_trace.csv
diagnostics.json
run_config_resolved.yaml
```

Current P6 status:

- `secondary_state_from_dc_initialization` converts P5
  `DCSecondaryInitialization` output into the P6 `SecondaryState` used by the
  no-IP/IP transient step kernels.
- This closes the pure-Python state handoff from DC initialization to TDEM
  stepping. DOLFINx operator assembly and time-loop wiring for the secondary
  solver are still pending.

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
python -m pytest -q tests/test_dolfinx_complex_terrain_leakage_forward.py
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

## Running Final No-IP/IP Acceptance Summary

After generating no-IP and IP validation artifact directories, write a final
combined gate report with:

```bash
tdem-ip-forward acceptance-report acceptance.yaml
```

Example config:

```yaml
acceptance:
  noip_summary_json: outputs/noip_3comp/error_summary.json
  ip_summary_json: outputs/ip_3comp/error_summary.json
  noip_diagnostics_json: outputs/noip_3comp/diagnostics.json
  ip_diagnostics_json: outputs/ip_3comp/diagnostics.json
  output_dir: outputs/final_acceptance
```

Outputs:

- `final_acceptance_summary.json`
- `final_acceptance_report.txt`

The command returns exit code `0` only if both no-IP and IP summaries have
`final_acceptance_passed=true`. It returns exit code `1` and records blocking
reasons when either case is missing or failed. When diagnostics paths are
provided, `final_acceptance_summary.json` also includes
`failure_diagnostics_by_case`, and the text report lists per-case diagnostic
reason codes and check statuses.
`corrected-model-run` writes `acceptance.yaml` as YAML text rather than JSON
text with a YAML extension.

## Writing Corrected-Model Case Specs

Use the canonical corrected-model helper to write no-IP/IP case metadata:

```bash
tdem-ip-forward corrected-model-spec outputs/corrected_model --output corrected_model_validation_spec.json
```

Use `--n-observation-times N` to override the default 80 log-spaced
observation times while keeping the same `1e-5 s` to `1 s` window:

```bash
tdem-ip-forward corrected-model-spec outputs/corrected_model --output corrected_model_validation_spec.json --n-observation-times 5
```

The spec records the corrected source/receiver coordinates, source current,
full observation time window, components, empymod primary configuration,
runner metadata, no-IP/IP material metadata, and per-case output directories.
It is intended as the shared input contract for the pending corrected-model
DOLFINx forward backend.

The pure orchestration command is:

```bash
tdem-ip-forward corrected-model-run corrected_model_validation_spec.json --case noip --output-root outputs/corrected_model
```

It writes the same validation artifact set as `validate-noip-3comp` and
`validate-ip-3comp` when supplied with working forward/reference runners. The
current default reference path uses `EmpymodPrimaryProvider`. The current
default DOLFINx forward path can run a small no-IP uniform-background
primary-secondary case by sampling `EmpymodPrimaryProvider` on the DOLFINx
Nedelec interpolation points and using the zero-secondary receiver projector.
This is a backend smoke, not a full corrected-model 5% acceptance run.
Each corrected-model run now also writes `model_schematic.png` beside the
response/error artifacts and records `diagnostics["model_schematic"]`.
The runner records `runtime_seconds.forward`, `runtime_seconds.reference`, and
`runtime_seconds.artifact_total` in `diagnostics.json` for each validation case.
For IP cases, the corrected-model reference path now converts the Prony/Debye
material metadata into empymod's Debye `res` dictionary, so the reference uses
the same `sigma_inf`, `delta_sigma_list`, and `tau_list` stored in the case
spec instead of silently falling back to the no-IP resistivity tuple.

## Published Paper Reproduction Target

The currently selected published model target is:

```text
title = Analysis of 3D induced polarization effects of SOTEM
journal = Journal of Applied Geophysics
volume = 233
publication_date = February 2025
article_number = 105613
article_id = S092698512400329X
doi = 10.1016/j.jappgeo.2024.105613
url = https://www.sciencedirect.com/science/article/pii/S092698512400329X
```

Public metadata/snippets align with the present SOTEM setup: a finite grounded
wire SOTEM source, `1000 m` transmitter length, `10 A` current, air conductivity
around `1e-6 S/m`, a background half-space around `0.01 S/m`, and a published
calculation domain of `4000 m x 4000 m x 1000 m`. The latest corrected
coordinates used here remain source `(-500, 200, -0.1) -> (500, 200, -0.1)` and
receiver `(0, -300, -0.1)`.

Publicly visible method/model metadata also indicate that the paper solves the
SOTEM response in the frequency domain with COMSOL, transforms to time domain
with frequency-time transformation, and discusses `Ex` and `Hz` responses for
polarized layers plus high-resistivity and low-resistivity polarized bodies.
These entries are recorded in `published_paper_model_target.json` as public
metadata only; they are not enough to reconstruct the full response curves.

Write the reproduction target metadata with:

```bash
tdem-ip-forward published-paper-model-spec dolfinx/runs/published_paper_target --output dolfinx/runs/published_paper_target/spec.json
```

This command records the paper identity, corrected source/receiver geometry,
full `1e-5 s` to `1 s` validation window, and the required comparison outputs.
It also explicitly lists the full-text parameters still required before an
actual paper-response overlay can be claimed:

```text
terrain_surface_or_layer_geometry
ip_anomaly_geometry
ip_anomaly_prony_or_cole_cole_parameters
all_receiver_locations_and_components
paper_plot_time_channels
digitized_or_tabulated_published_response_values
```

Current status: this is a reproducibility target definition only. It is not yet
a completed reproduction of the published response curves, because the full
paper model tables/figure values have not been extracted into the run spec.
The public pages checked so far expose enough information to identify the
target and coarse setup, but not enough to digitize the published response
curves or define the full 3D IP anomaly. The next reproducibility step is to
extract the missing model table/figure data from the full article text or a
user-supplied PDF, then add those values to this spec and run the overlay.

## Corrected-Scale Leakage Diagnostic Spec

Write a memory-safe corrected-scale leakage-channel diagnostic spec with:

```bash
tdem-ip-forward corrected-leakage-model-spec dolfinx/runs/corrected_leakage_model --output dolfinx/runs/corrected_leakage_model/spec.json --n-observation-times 3
```

The generated no-IP/IP specs keep the corrected latest source and receiver,
cover the full `1e-5 s` to `1 s` observation window, and add DOLFINx forward
metadata:

```text
domain_min = [-2000, -2000, -1000]
domain_max = [2000, 2000, 100]
cells = [2, 2, 1]
leakage_channel = four-point irregular polyline
```

A WSL no-IP diagnostic run completed with:

```bash
PYTHONPATH=src /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m atem3d.cli corrected-model-run dolfinx/runs/corrected_leakage_model/spec.json --case noip --output-root dolfinx/runs/corrected_leakage_model_run
```

Output directory:

```text
dolfinx/runs/corrected_leakage_model_run/noip_3comp
```

Artifacts written:

```text
model_schematic.png
predictions.csv
reference_empymod_or_1d.csv
errors.csv
error_summary.json
comparison_3comp.png
error_curves_3comp.png
diagnostics.json
run_config_resolved.yaml
```

Runtime:

```text
forward = 34.164 s
reference = 0.443 s
artifact_total = 1.019 s
```

The run reports `final_acceptance_passed=false` with blocking reasons
`validation_scope_not_corrected_model_full` and `physical_error_gate_failed`.
This is expected: the prediction includes a 3D leakage-channel anomaly, while
the reference is the empymod background response. Therefore this result proves
the corrected-scale leakage diagnostic backend/artifact path runs, not that the
3D anomaly response has an accepted reference-level error.

The model schematic can be regenerated with:

```bash
tdem-ip-forward model-schematic dolfinx/runs/corrected_leakage_model/spec.json --case noip --output dolfinx/runs/corrected_leakage_model_run/noip_3comp/model_schematic.png
```

The schematic is a geometry artifact only. It does not affect the forward solve
or error metrics.

A WSL IP diagnostic run using the same spec also completed with:

```bash
PYTHONPATH=src /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m atem3d.cli corrected-model-run dolfinx/runs/corrected_leakage_model/spec.json --case ip --output-root dolfinx/runs/corrected_leakage_model_run
```

Output directory:

```text
dolfinx/runs/corrected_leakage_model_run/ip_3comp
```

The IP model schematic can be regenerated with:

```bash
tdem-ip-forward model-schematic dolfinx/runs/corrected_leakage_model/spec.json --case ip --output dolfinx/runs/corrected_leakage_model_run/ip_3comp/model_schematic.png
```

Runtime:

```text
forward = 35.159 s
reference = 0.435 s
artifact_total = 0.965 s
```

The IP artifact summary records the Prony background metadata:

```text
sigma0 = 0.01 S/m
sigma_inf = 0.012 S/m
delta_sigma_list = [0.002]
tau_list = [0.1]
prony_dc_constraint_error = 0.0
```

It also reports `final_acceptance_passed=false` with the same expected
diagnostic blocking reasons. The weak `Ey` component is handled by the weak
component policy; the physical failed component is `Ex`. This confirms the
corrected-scale no-IP and IP leakage-channel diagnostic artifact paths both
run under WSL on the 32 GB machine, while final 3D anomaly accuracy still
requires a defensible 3D reference or published response data.

Dedicated IP-minus-noIP polarization-effect artifacts can be written with:

```bash
tdem-ip-forward polarization-effect dolfinx/runs/corrected_leakage_model_run/noip_3comp dolfinx/runs/corrected_leakage_model_run/ip_3comp --output-dir dolfinx/runs/corrected_leakage_model_run/polarization_effect
```

The command writes:

```text
polarization_effect_predictions.csv
polarization_effect_reference.csv
polarization_effect_errors.csv
polarization_effect_summary.json
polarization_effect_comparison.png
polarization_effect_error_curves.png
```

For the current corrected-scale leakage diagnostic, the polarization-effect
summary reports `definition=ip_minus_noip`, components `Ex`, `Ey`, and
`dBzdt`, and `pass_all_components=false`. The physical failed components are
`Ex` and `dBzdt`; weak `Ey` is handled by the weak-component policy. This is
again a diagnostic artifact because both no-IP/IP responses include the 3D
leakage channel while the reference effects are computed from the empymod
background responses.

A WSL CLI smoke completed under:

```text
dolfinx/runs/corrected_model_noip_smoke/noip_3comp
```

It used two observation times (`1e-5 s` and `1 s`), a tiny DOLFINx box mesh
with one cell per axis, `EmpymodPrimaryProvider(srcpts=3)`, and the no-IP
uniform-background zero-secondary path. It wrote the required CSV/JSON/PNG
artifact set and reported zero error against empymod because the total response
is exactly the primary response in this uniform no-IP smoke. This verifies the
CLI/artifact/backend plumbing only; it does not validate nonzero contrast,
terrain/leakage, IP memory, or a production-resolution full-time run.

A local two-time IP empymod reference smoke using the corrected Prony metadata
also returned finite `Ex/Ey/dBzdt` values. This validates the IP reference
plumbing only; it does not validate the DOLFINx IP forward response.

A later WSL CLI smoke completed both no-IP and IP cases under:

```text
dolfinx/runs/corrected_model_noip_ip_smoke
```

This smoke used the same two observation times, tiny box mesh, and
`EmpymodPrimaryProvider(srcpts=3)`. For IP, the Debye/Prony background is
included in the empymod primary provider and the DOLFINx secondary field is
zero because there is no 3D material contrast in this smoke. Both no-IP and IP
artifact summaries report `final_acceptance_passed=true` with zero error, but
this is a plumbing/primary-background consistency check only. It is not the
production corrected-model full-window run with a resolved mesh, terrain,
leakage channel, or 3D IP anomaly.
When both cases are run, `corrected-model-run` now writes
`acceptance.yaml`, and `tdem-ip-forward acceptance-report` can consume it to
write `final_acceptance/final_acceptance_summary.json` and
`final_acceptance_report.txt`. The same two-time WSL smoke exercised this
chain successfully.
When both cases are run, `corrected-model-run` also writes
`polarization_effect/` using the no-IP/IP case artifact directories, so the
IP-minus-noIP response and error plots are regenerated automatically.

Integrated corrected-scale leakage WSL run:

```bash
PYTHONPATH=src /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m atem3d.cli corrected-model-run dolfinx/runs/corrected_leakage_model/spec.json --case both --output-root dolfinx/runs/corrected_leakage_model_both_integrated
```

This run wrote:

```text
dolfinx/runs/corrected_leakage_model_both_integrated/noip_3comp
dolfinx/runs/corrected_leakage_model_both_integrated/ip_3comp
dolfinx/runs/corrected_leakage_model_both_integrated/acceptance.yaml
dolfinx/runs/corrected_leakage_model_both_integrated/polarization_effect
```

The no-IP and IP case directories both contain `model_schematic.png`, and
their `diagnostics.json` files record:

```text
source_length_m = 1000.0
parallel_offset_m = 500.0
domain_extent_m = [4000.0, 4000.0, 1100.0]
leakage_point_count = 4
```

Runtime:

```text
no-IP forward = 34.298 s, reference = 0.436 s, artifact_total = 1.128 s
IP    forward = 31.207 s, reference = 0.394 s, artifact_total = 0.733 s
```

As expected, both case summaries still report `final_acceptance_passed=false`
with `validation_scope_not_corrected_model_full` and
`physical_error_gate_failed`: this is a 3D leakage-channel diagnostic against
a background empymod reference, not a final 3D anomaly accuracy reference.

The combined acceptance report for this integrated diagnostic was generated
with:

```bash
python -m atem3d.cli acceptance-report dolfinx/runs/corrected_leakage_model_both_integrated/acceptance.yaml
```

It wrote:

```text
dolfinx/runs/corrected_leakage_model_both_integrated/final_acceptance/final_acceptance_summary.json
dolfinx/runs/corrected_leakage_model_both_integrated/final_acceptance/final_acceptance_report.txt
```

Result:

```text
FINAL_ACCEPTANCE_PASSED=false
failed_cases=noip,ip
noip blocking = validation_scope_not_corrected_model_full, physical_error_gate_failed
ip blocking = validation_scope_not_corrected_model_full, physical_error_gate_failed
```

This confirms the final gate blocks diagnostic leakage runs from being
misreported as accepted task-book validation.

An 80-observation-time WSL smoke also completed under:

```text
dolfinx/runs/corrected_model_noip_ip_80pt_smoke
```

It used `1e-5 s <= t_obs <= 1 s`, the same tiny one-cell-per-axis DOLFINx box,
and `EmpymodPrimaryProvider(srcpts=3)`. Both no-IP and IP artifact summaries
reported `final_acceptance_passed=true`, zero component error, and the final
acceptance report passed. Recorded runtimes were:

```text
no-IP forward 214.619 s, reference 10.690 s, artifact 0.981 s
IP    forward 207.851 s, reference 10.777 s, artifact 0.700 s
```

This validates the 80-point corrected-model orchestration, plotting, runtime
diagnostics, and final-acceptance plumbing for a primary-background smoke only.
It still does not validate a resolved production mesh, 3D contrast, terrain,
leakage channel, or a DOLFINx-computed IP secondary anomaly.

The corrected-model DOLFINx forward runner now also accepts
`dolfinx_forward.secondary_sigma` for a no-IP nonzero-conductivity-contrast
smoke. A WSL test verifies that `secondary_sigma=0.02 S/m` with a
`0.01 S/m` background produces a finite response that differs from the empymod
background reference. This exercises the DOLFINx secondary adapter through the
corrected runner, but still on a tiny synthetic box rather than a resolved
production terrain/leakage mesh.

The corrected-model runner also accepts a `dolfinx_forward.leakage_channel`
block with channel points, radius, and leakage conductivity. This path marks
cells with `apply_leakage_channel_marker`, builds a `CellMaterialMap`, converts
it with `_make_dolfinx_materials_from_cell_material_map`, and runs the same
primary-secondary DOLFINx operator. A WSL smoke verifies a finite response that
differs from the empymod background reference. This connects the leakage-channel
material-map machinery to the corrected runner, though still on a tiny box
rather than a production terrain mesh.

The same leakage-channel block now also supports Prony/Debye IP material
metadata through `sigma_inf`, `delta_sigma_list`, and `tau_list`. A WSL smoke
verifies that an IP leakage channel runs through the spatial-DG0 Debye material
path and produces a finite response different from the empymod background
reference. This exercises IP secondary-anomaly plumbing in the corrected runner
on a tiny synthetic box.

An IP leakage artifact smoke completed under:

```text
dolfinx/runs/corrected_model_ip_leakage_smoke/ip_3comp
```

It wrote the full CSV/JSON/PNG artifact set for two observation times. Because
the reference is still the 1D/background empymod response while the prediction
contains a 3D IP leakage anomaly, the validation correctly failed the physical
gate with `physical_error_gate_failed`. Recorded runtime was:

```text
IP leakage forward 22.822 s, reference 0.291 s, artifact 0.987 s
```

The summary reported `max_peak_normalized_error_Ex=0.0088655`,
`max_peak_normalized_error_dBzdt=4.3165e-06`, and a weak near-zero `Ey`
component handled by the weak-component policy. This is an artifact/diagnostic
smoke for 3D IP anomaly output, not an acceptance comparison against a 3D
reference.

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

- Half-strength conductivity divergence-cleaning diagnostic:
  - Directory:
    `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_substep4_divclean05_growth2_diag`.
  - Code change: `--divergence-cleaning-strength` now scales the
    conductivity cleaning correction. Default `1.0` preserves the previous
    full projection.
  - Diagnostic change relative to the full-strength coarse run:
    `--divergence-cleaning-strength 0.5`.
  - Execution:
    - First WSL segment reached `t_obs=0.32768 s` and then hit KSP
      `reason=-3` at the next step despite a small residual; resume with
      `--max-it 10000` completed to `t_obs=1.0 s`.
    - WSL was shut down after both segments.
  - Result:
    - `pass_all_components = false`
    - `physical_pass_all_components = false`
    - `physical_failed_components = ["Ex", "Hz", "dBzdt"]`
    - `weak_component_passed = true`
    - `weak_component_scaled_abs_error_max(Ey) = 0.023853142889896116`
    - `max_error_Ex = 0.9496837918444295` at `t_obs=0.04096 s`
    - `max_peak_normalized_error_Ex = 0.1263291999451112`
    - `max_error_dBzdt = 3.6630024137970363` at `t_obs=0.04096 s`
    - `max_peak_normalized_error_dBzdt = 0.0863076672288047`
  - Interpretation:
    - Half-strength repeated every post-ramp step behaves almost the same as
      full-strength cleaning over this coarse time grid. It still removes the
      late static offset, but does not recover the mid-time `Ex/dBzdt`
      amplitude.
    - Simple per-step scaling is therefore not the right final control
      mechanism. The next diagnostic should test delayed/thresholded cleaning
      or a variational divergence-control term that damps static residuals
      without repeatedly projecting away the inductive response.

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

## P2 Delayed Divergence-Cleaning Diagnostic

After adding persistent divergence-cleaning solver-log fields, I added a
diagnostic gate:

```text
--divergence-cleaning-t-obs-min <seconds>
```

The default is `0`, which preserves the previous behavior. A positive value
delays conductivity divergence cleaning until the post-ramp observation time
exceeds the configured threshold.

The validation artifact writer now also records a `divergence_cleaning`
summary in `diagnostics.json` when `solver_log` contains cleaning records,
including the cleaned step count, first cleaning time, maximum pre-clean
residual, and maximum correction norm.

Two WSL diagnostics were attempted with the corrected latest model geometry:

- Fine run:
  `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_substep4_divclean_delay008_growth2_diag`
  used `source_mesh_size=40`, `receiver_mesh_size=20`,
  `time_growth=2`, and `divergence_cleaning_t_obs_min=0.08`. It timed out
  after 1 hour at `t_obs=0.04096 s`, before the cleaning gate was reached.
  This run is not a validation result.
- Coarse diagnostic:
  `dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_substep4_divclean_delay002_growth2_coarse_diag`
  used `source_mesh_size=80`, `receiver_mesh_size=40`,
  `time_growth=2`, and `divergence_cleaning_t_obs_min=0.02`. The first WSL
  segment reached `t_obs=0.65536 s` after the 30 minute timeout; a resume
  completed the final `1 s` output and wrote the full artifact set. The resume
  report records `total=22.646 s`, `forward solve=10.274 s`, and
  `empymod reference=1.822 s`.

The coarse delayed-cleaning result does not meet the 5% target:

```text
max_error_Ex = 2.1758013524216264
max_peak_normalized_error_Ex = 0.1816751111359116
max_error_dBzdt = 3.4342239301412913
max_peak_normalized_error_dBzdt = 0.0977590474346503
physical_failed_components = Ex, Hz, dBzdt
weak_component_passed = true
```

Key response comparison:

```text
t_obs=1 s:
  Ex_pred = 2.5184030602385072e-08
  Ex_ref  = 2.1069619302160945e-08

t_obs=0.08192 s:
  dBzdt_pred = 9.0908513912550654e-11
  dBzdt_ref  = 2.0501561343036178e-11
```

Interpretation: delaying the projection can remove the late static electric
offset, but the first cleaning step applies a very large correction
(`div_clean_before=1.173273e+04`, `correction=1.870250e+02` at
`t_obs=0.02048 s`) and the mid-time `dBzdt` response remains over-amplified.
This supports the current root-cause diagnosis: the total-field E-form state
contains a gradient/static residual, but a post-step projection is too
aggressive for the inductive transient. The next P2 path should be a
variational divergence-control term or a primary-secondary formulation, not
more per-step projection tuning.

## P2 Variational Divergence-Control Smoke

I added an optional implicit weak-divergence control term for the E-form
total-field solver:

```text
--divergence-control-weight <float>
--divergence-control-t-obs-min <seconds>
```

The default weight is `0`, so existing runs are unchanged. When enabled for a
non-polarizable run, the solver builds a sparse diagnostic penalty matrix

```text
M_sigma G G^T M_sigma
```

and adds `divergence_control_weight * (M_sigma G G^T M_sigma)` to the left-hand
side after the configured post-ramp observation time. This is intended as a
variational alternative to post-step projection, not as an accepted final
formulation yet.

Smoke run:

```text
dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_divcontrol_smoke
```

Configuration summary:

```text
source_mesh_size = 120 m
receiver_mesh_size = 80 m
t_obs = 1e-5, 2e-5, 4e-5 s
divergence_control_weight = 1e-12
divergence_control_t_obs_min = 1e-5 s
```

The WSL run completed and shut down successfully. The report records:

```text
total runtime = 77.648 s
forward solve = 68.329 s
empymod reference = 0.986 s
```

The short-window smoke does not pass the 5% gate:

```text
max_error_Ex = 0.5210836011038802
max_peak_normalized_error_Ex = 0.5198399885248545
max_error_dBzdt = 0.3560966799435061
max_peak_normalized_error_dBzdt = 0.3549869046652062
physical_failed_components = Ex, dBzdt
weak_component_passed = true
```

Interpretation: the variational matrix path is operational and within the
32 GB memory budget on a coarse smoke mesh, but the tested weight does not
solve the early-time error. A meaningful sweep needs normalized scaling of the
penalty relative to the mass/stiffness terms before spending a full-window
fine run.

## P2 LHS-Scaled Divergence-Control Diagnostics

I added an explicit divergence-control scale mode:

```text
--divergence-control-scale absolute|mass|stiffness|lhs
```

The default remains `absolute`, preserving existing runs. The new `lhs` mode
treats `divergence_control_weight` as a dimensionless fraction of the current
implicit left-hand operator norm:

```text
applied_weight = weight * (|lhs_mass| ||M_sigma|| + |lhs_stiffness| ||K||)
                 / ||M_sigma G G^T M_sigma||
```

The solver now records the applied divergence-control coefficient, matrix
norm, reference LHS norm, relative weight, and scale mode in solver logs,
partial/checkpoint NPZ files, `diagnostics.json`, `run_config_resolved.yaml`,
and the text report. This makes parameter sweeps auditable instead of relying
on an opaque absolute coefficient.

Smoke run:

```text
dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_divcontrol_lhs_smoke
```

Configuration summary:

```text
t_obs = 1e-5, 2e-5 s
source = (-500, 200, -0.1) -> (500, 200, -0.1)
receiver = (0, -300, -0.1)
source_mesh_size = 180 m
receiver_mesh_size = 120 m
divergence_control_weight = 1e-3
divergence_control_scale = lhs
```

The WSL run completed and shut down successfully. The diagnostics record:

```text
applied_step_count = 2
first_applied_observation_time = 1e-5 s
max_reference_norm = 5.080943524505622e9
max_matrix_norm = 4.467403547193896e5
max_applied_weight = 11.373370394750006
max_relative_weight = 0.001
scale_values = lhs
```

Interpretation: the normalized scaling is now functioning on the real PETSc
matrix path, and the recorded relative weight matches the configured
dimensionless `1e-3`. This smoke is not an accuracy acceptance run; the coarse
two-point run still exceeds the physical gate and only validates the new
diagnostic/control plumbing.

## P2 LHS Divergence-Control Short-Window Sweep

I ran a controlled short-window no-IP sweep using the same corrected latest
model and the same coarse mesh settings as the `lhs` smoke:

```text
source = (-500, 200, -0.1) -> (500, 200, -0.1)
receiver = (0, -300, -0.1)
t_obs = 1e-5, 2e-5 s
source_mesh_size = 180 m
receiver_mesh_size = 120 m
divergence_control_scale = lhs
relative weights = 0, 1e-5, 1e-4, 1e-3
```

Summary artifacts:

```text
dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_divcontrol_lhs_sweep_summary
```

The sweep table is:

```text
relative_weight  max_peak_error_Ex  max_peak_error_dBzdt  max_applied_weight
0                0.6724526431       0.6486769606          0
1e-5             0.6724526438       0.6486769648          0.113733704
1e-4             0.6724526471       0.6486769711          1.137337039
1e-3             0.6724526775       0.6486769895          11.373370395
```

Interpretation: on this controlled short-window mesh, the normalized
divergence-control term does not reduce the early Ex or dBzdt disagreement
with empymod. Increasing the relative weight slightly worsens the metrics. This
is evidence that the dominant early-time discrepancy is not primarily fixed by
weak divergence damping. The next useful P2 work should focus on source/receiver
consistency and the primary-secondary path instead of continuing to tune this
variational penalty.

## P2 Receiver Evaluation Mode Audit

I ran a controlled receiver sampling audit on the same corrected latest model
and short-window coarse mesh. The audit keeps the source, mesh, time stepping,
boundary, and no-IP physics fixed and changes only:

```text
--receiver-evaluation-mode first_cell|mean|median|nearest_center|shallowest
```

Summary artifacts:

```text
dolfinx/current_task_runs/y200_rxminus300_noip_meshseg_analyticdc_receiver_mode_audit_summary
```

The peak-normalized error table is:

```text
receiver_mode   Ex error      dBzdt error
median          0.6724526431  0.6486769606
mean            0.6786600536  0.5908231256
nearest_center  0.6723799271  0.4377120033
shallowest      0.6723799271  0.4377120033
first_cell      0.6725253592  0.4411768325
```

Interpretation: receiver cell-candidate selection materially changes dBzdt,
reducing the short-window dBzdt error from about 65% to about 44% for
`nearest_center`/`shallowest`. However, Ex remains near 67% for all receiver
modes. Therefore receiver sampling is part of the magnetic-derivative issue,
but it is not the dominant source of the early electric-field disagreement.
The remaining Ex error points back to source loading, primary/background
field consistency, or the total-field formulation.

## P2 Magnetic Receiver Mode Audit

I ran a controlled one-output WSL audit on the corrected latest geometry using
the same mesh/time/source settings and changing only:

```text
--magnetic-receiver-mode biot_current|biot_ohmic|faraday_integrated
```

The output directories are intentionally under ignored `dolfinx/runs/`:

```text
dolfinx/runs/y200_rxminus300_magnetic_mode_biot_current_onepoint
dolfinx/runs/y200_rxminus300_magnetic_mode_biot_ohmic_onepoint
dolfinx/runs/y200_rxminus300_magnetic_mode_faraday_integrated_onepoint
```

All three runs used the corrected model consistency checks:

```text
source: (-500, 200, -0.1) -> (500, 200, -0.1)
receiver: (0, -300, -0.1)
expected source length: 1000 m
expected parallel offset: 500 m
stop-after-outputs: 1
```

First-output results at `t_obs=1e-5 s`:

```text
mode                 Hz pred          Hz ref           Hz peak error   dBzdt peak error   Ex peak error
biot_current         -2.2265429e-3    -2.1976186e-3    1.316%          23.317%            44.154%
biot_ohmic           -2.2265429e-3    -2.1976186e-3    1.316%          23.317%            44.154%
faraday_integrated   -2.2201369e-3    -2.1976186e-3    1.025%          23.317%            44.154%
```

Interpretation: `biot_current` and `biot_ohmic` are identical at this first
after-ramp output for this non-polarizable one-point audit. The new
`faraday_integrated` receiver mode runs in WSL and gives a slightly better
`Hz` value than Biot-Savart recovery, but the dominant errors are unchanged:
`Ex` is about `44%` and `dBzdt` is about `23%`. This supports the current
diagnosis that the remaining short-window failure is not mainly a Biot `Hz`
recovery problem; it is tied to electric/source/receiver-curl consistency in
the total-field E-form path.

## P2 Initial Field Curl Diagnostic

I added a runtime diagnostic for the DC/on-time initial electric field:

```text
diagnostics["initial_field"]["quantity"] = "-curl(E_initial)"
diagnostics["initial_field"]["initial_curl_residual"]
diagnostics["initial_field"]["initial_curl_max_abs"]
diagnostics["source_consistency"]["initial_curl_residual"]
```

A small WSL smoke run using the corrected default geometry completed under:

```text
dolfinx/analyticdc_small_smoke
```

The smoke used a deliberately small finite box and `--initial-dc-mode
analytic_halfspace`, so it is not an accuracy claim. It verifies the DOLFINx
runtime path and produced:

```text
initial_curl_residual = 1334.1011123823428
initial_curl_max_abs  = 1321.9773822219981
```

Interpretation: the new diagnostic is active and can expose whether the
interpolated/initialized DC field is curl-free in the discrete receiver/FEM
spaces. The large value in this small analytic-halfspace smoke is a warning
that the initial-field interpolation/source transition must be audited before
treating late/early electric-field errors as only boundary or receiver issues.

I then reran the same small model with `--initial-dc-mode fem` and changed no
other physics/model setting. The FEM DC initializer produced a near-zero
discrete initial curl:

```text
initial_dc_mode      initial_curl_residual    initial_curl_max_abs     Ex peak error   dBzdt peak error
analytic_halfspace   1.334101112e3           1.321977382e3            58.725%         46.797%
fem                  2.193462260e-13         7.352684444e-14          104.777%        45.776%
```

Interpretation: the large `initial_curl_residual` is specific to the analytic
halfspace field interpolation path on this small mesh; the FEM scalar-potential
initializer is discretely curl-free. However, the first-output `Ex` and
`dBzdt` errors remain large in both modes, so the total-field source/DC
transition and early transient source loading still need deeper treatment.

## P2 Source Term Mode Audit

I ran a small WSL one-output source-term audit using the same finite box,
corrected default geometry, and FEM DC initializer. The audit changed only the
transient source loading mode:

```text
initial_dc_mode = fem
t_obs = 1e-5 s
source_term_mode = impressed_current | primary_dc
```

The comparison is:

```text
source_term_mode   initial_curl_residual  Ex pred          Ex ref           Ex peak error   dBzdt pred       dBzdt ref       dBzdt peak error
impressed_current  2.193462260e-13        -8.5300569e-05   1.7857499e-03    104.777%        4.8722848e-06   8.9855492e-06   45.776%
primary_dc         2.193462260e-13        -1.8242988e-02   1.7857499e-03    1121.587%       7.3485085e-09   8.9855492e-06   99.918%
```

Both runs preserve the source consistency checks:

```text
source_endpoint_balance_residual = 5.765754464368252e-11
waveform_integral_residual = 0.0
endpoint_source_total_sum = 0.0
```

Interpretation: replacing the impressed-current turn-off source with the
current `primary_dc` source term is not an improvement in the total-field
solver. It keeps the FEM DC initial field curl-free, but it over-amplifies `Ex`
and almost removes the inductive `dBzdt` response at the first output. This
supports the current diagnosis that a correct primary-secondary formulation
needs explicit primary-field/background-current accounting, not a simple
source-term substitution inside the existing total-field equation.

## Known Limitations

- `diagnose_source_consistency` currently reports waveform-integral and endpoint-total checks without full FEM matrix residuals unless a source projection residual is provided.
- `source_term_mode=primary_dc` is currently diagnostic only. In a small WSL
  one-output audit with `initial_dc_mode=fem`, it worsened the first-output
  `Ex` and `dBzdt` errors relative to `impressed_current`, so it must not be
  treated as the accepted primary-secondary implementation.
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
- E-form receiver diagnostics now persist both `dBzdt_curl` from `-curl(E)`
  and `dBzdt_biot_rate` from finite-difference Biot-Savart `Hz` when a Biot
  magnetic receiver mode is active. This is a diagnostic split only; it does
  not by itself establish full-window 5% agreement.
- Receiver point cell-candidate selection affects short-window `dBzdt`
  substantially on the corrected latest model, but all tested selection modes
  leave `Ex` near 67% peak-normalized error, so receiver sampling alone does
  not fix the early no-IP validation failure.
- Faraday-integrated receiver `Hz` recovery is implemented as an E-form
  diagnostic output mode. It still needs a controlled WSL validation run on the
  corrected latest model before it can be used in an accuracy claim.
- Late-diffusion diagnostics now report finite-domain coverage separately from
  the local refinement box. Existing artifacts generated before this fix may
  still show the old ambiguous `actual radius/depth` wording until
  regenerated with `--postprocess-partial`.
- P3 currently provides the material API and memory-update tests; DOLFINx total-field IP assembly still needs to be migrated to this API and verified against no-IP when `delta_sigma=0`.
- P4 currently provides zero/cached primary providers, receiver-side empymod primary sampling, runner-backed FEM point `E_p(t)` sampling, injected DC primary point sampling, a default uniform-halfspace analytic grounded-wire DC fallback, a provider-to-FEM-point interpolation adapter, a DOLFINx-style tabulated callable assembler, a shared DOLFINx Nedelec callable interpolation helper, and a DOLFINx operator helper that samples primary providers on exported physical Nedelec interpolation points. Corrected-model primary-provider validation remains pending.
- The default `EmpymodPrimaryProvider` runner now covers transient empymod
  primary samples and uniform-halfspace analytic DC primary samples. A layered
  or complex-background DC primary still requires an explicit `dc_runner`, so
  the full corrected-model primary-secondary validation runner remains pending.
- P5 currently provides a pure initialization core with an injected secondary field solver, a provider-driven entry point that consumes `E_p,dc` samples, and a DOLFINx scalar secondary-potential solver with WSL zero-contrast and nonzero-contrast unit-cube smoke tests. Integration into a full corrected-model DOLFINx primary-secondary run remains pending.
- P6 currently provides pure no-IP/IP time-step kernels with injected secondary solvers, a DC-initialization-to-transient-state bridge, a pure primary-secondary forward orchestration core, a reusable secondary receiver projection adapter, a DOLFINx-backed secondary step solver with WSL zero-RHS and nonzero constant-RHS PETSc smoke tests, DOLFINx primary-secondary zero-contrast and uniform nonzero-contrast forward smokes, a variable-DG0 no-IP DOLFINx primary-secondary state-stepper smoke, a scalar Debye/Prony IP DOLFINx state-stepper smoke, a spatial-DG0 `delta_sigma` IP smoke using `debye["delta_functions"]`, physical Nedelec interpolation-point export for non-constant tabulated primary/RHS fields, and a DOLFINx operator helper that wires primary-provider FEM sampling into the primary-secondary operator. Corrected-model validation and full no-IP/IP 5% acceptance remain pending.
- P7 currently verifies artifact generation from supplied arrays and the
  corrected-model runner scaffold with injected runners. A WSL smoke now runs
  the default DOLFINx no-IP uniform-background forward backend on a tiny mesh;
  the IP empymod reference path now uses the corrected Prony/Debye material
  metadata, and a two-time WSL CLI smoke writes both no-IP/IP artifact sets.
  These do not yet prove production full-window 5% physical agreement for the
  corrected model.
- P7 CLI reads precomputed prediction/reference CSV files for
  `validate-noip-3comp` and `validate-ip-3comp`. `corrected-model-run` now
  exists as the orchestration entry point; its default DOLFINx forward runner
  is currently smoke-tested for no-IP zero-secondary cases only.
- `tdem-ip-forward acceptance-report` summarizes no-IP/IP validation outputs;
  it does not create those outputs or repair failing component errors. When
  diagnostics paths are supplied, it also carries per-case validation-failure
  reason codes and check statuses into the combined final report.
- Validation artifact diagnostics now include task-book failure reason codes
  and seven structured follow-up checks: time step, mesh, boundary, source
  term, receiver sampling, magnetic recovery, and IP memory.
- Validation artifact `run_config_resolved.yaml` files are now emitted as
  YAML text and remain readable by `yaml.safe_load`; writing does not require
  `PyYAML` in the runtime environment.
- Corrected-model orchestration `acceptance.yaml` files are now emitted as
  YAML text and remain readable by `yaml.safe_load`; writing does not require
  `PyYAML` in the runtime environment.
- WSL DOLFINx smoke regression after the YAML fallback fix:
  `PYTHONPATH=src /home/paidaxin/miniconda3/envs/fenicsx/bin/python -m pytest -q tests/test_dolfinx_primary_secondary_forward_smoke.py tests/test_dolfinx_complex_terrain_leakage_forward.py`
  completed with `13 passed`. WSL was then shut down and `Ubuntu Stopped` was
  confirmed.
- WSL PyYAML-free YAML read regression:
  a fallback-written `acceptance.yaml` was read through `atem3d.cli._load_yaml`
  in `/home/paidaxin/miniconda3/envs/fenicsx/bin/python` and printed
  `fallback_yaml_load_ok`. WSL was then shut down and `Ubuntu Stopped` was
  confirmed.
- WSL PyYAML-free task-book inline-list YAML regression:
  fallback loading of `source.start: [-500.0, 200.0, -0.1]` and
  `validation.components: [Ex, Ey, dBzdt]` completed in the same FEniCSx
  environment and printed `fallback_inline_list_yaml_ok`. WSL was then shut
  down and `Ubuntu Stopped` was confirmed.
- The PyYAML-free YAML reader now also handles task-book style nested inline
  lists such as `time_steps: [[0.001, 3], 0.005]` and sequence mappings such
  as `receivers: - id: rx1 ...` with following indented fields. Windows
  verification ran `python -m pytest -q` with all tests passing (`2 skipped`).
  A WSL FEniCSx regression printed `wsl_fallback_nested_yaml_ok`; WSL was then
  shut down and `Ubuntu Stopped` was confirmed.
- `tdem-ip-forward corrected-model-spec` writes canonical corrected-model
  metadata, including runner/material metadata, and accepts
  `--n-observation-times` for memory-safe reduced diagnostic specs.
  `tdem-ip-forward corrected-model-run` can write artifacts with injected
  runners and has a small WSL-tested no-IP default DOLFINx forward path.
  Nonzero-contrast/IP corrected-model forward validation remains pending.
- `atem3d-validate-empymod --artifact-dir` bridges real validation results to artifact files, but final 5% agreement still depends on the underlying simulation/reference result.
- P8 currently verifies marker/material/channel geometry utilities, runs a
  small DOLFINx primary-secondary leakage-channel forward smoke on a unit-cube
  mesh with DG0 material markers, and runs a generated Gmsh terrain/leakage
  mesh through the same primary-secondary forward path. A corrected-model-scale
  terrain/leakage example remains pending.
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
- `divergence_cleaning_strength` is currently a diagnostic scaling factor.
  Applying a half correction at every post-ramp step behaves nearly like full
  cleaning over the coarse full-window run, so it does not solve the P2
  full-window error.
- `divergence_cleaning_t_obs_min` is a diagnostic gate. The coarse
  `t_obs_min=0.02 s` run still fails the P2 physical gate, so delayed
  post-step cleaning should not be treated as the accepted algorithm.
- `divergence_control_weight` is implemented only as a diagnostic weak
  divergence-control term for non-polarizable E-form runs. The `lhs` scale mode
  now makes the weight dimensionless relative to the implicit LHS matrix, but
  it is still a diagnostic path and does not meet the P2 5% gate. A controlled
  short-window sweep over relative weights `0..1e-3` did not improve Ex/dBzdt.
- `compute_error` and the validation artifact writer now share the task-book
  floor policy. Older reports generated before this change should be
  regenerated with `--postprocess-partial` before comparing error numbers.

## Next Steps

1. Replace the current post-ramp conductivity divergence projection with a
   variational consistency treatment or move to the primary-secondary solver
   path. Simple per-step strength scaling and delayed post-step cleaning have
   both been shown to be insufficient.
   The new divergence-control path now has dimensionless `lhs` scaling; the
   short-window sweep did not improve the early error. Receiver sampling mode
   audits improved dBzdt but not Ex, so the next step should shift to source
   loading/DC-primary consistency and the primary-secondary path before any full
   fine-grid run.
2. Keep mesh-segment line-source integration as the current source baseline,
   and add an explicit de Rham/source-edge orientation audit before replacing
   it with any DOLFINx-native source assembly.
3. Do not continue tuning `source_term_mode=primary_dc` as a total-field
   shortcut. Move primary/background-current accounting into the real
   primary-secondary path.
4. Run a controlled WSL comparison of `biot_current`, `biot_ohmic`, and
   `faraday_integrated` magnetic receiver modes using the persisted
   `dBzdt_curl`/`dBzdt_biot_rate` receiver diagnostics to localize the
   magnetic-rate recovery error.
5. Continue P3 by wiring `PronyConductivity` into DOLFINx total-field IP assembly and adding solver-level `delta_sigma=0` no-IP equivalence tests.
6. Continue P4/P5 by wiring the `PrimaryFEMInterpolator`/`TabulatedVectorField` adapter through `_interpolate_vector_callable_to_nedelec_function` for primary-secondary DC initialization and time stepping.
7. Continue P5 by wiring the tested DOLFINx scalar DC secondary solve into a
   full corrected-model primary-secondary run.
8. Continue P6 by wiring the step kernels to DOLFINx FEM operators and receiver operators.
9. Continue P7 by connecting `validation_3comp` to real no-IP/IP empymod or 1D reference runs over `1e-5 s <= t_obs <= 1 s`.
10. Continue P8 by scaling the generated Gmsh terrain/leakage smoke toward the
    corrected source/receiver geometry and writing full validation artifacts.
