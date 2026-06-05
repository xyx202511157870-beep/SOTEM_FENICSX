# P0-P3 Implementation Report

## Scope

This report covers the implementation rounds currently committed or staged from the task book:

- P0: freeze the current total-field baseline.
- P1: add waveform interfaces, full turn-off time grid, and interval-average `dI/dt`.
- P2: add no-IP three-component validation artifacts and diagnostics.
- P3 partial: add a Debye/Prony material conductivity API and pure-Python memory tests.

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

- `dolfinx/legacy_total_field_baseline.py`
  - Frozen copy of the current total-field baseline implementation.

- `src/atem3d/materials/prony.py`
  - `DebyeTerm`
  - `PronyConductivity`
  - Backward-Euler `alpha`, `beta`, and `sigma_eff`.
  - `chi_k` memory initialization/update helpers.
  - `J = sigma_inf E - sum(delta_sigma_k chi_k)` current-density helper.
  - Conversion to/from the existing `atem3d.ip.DebyeIPModel` for solver migration.

## Tests Added

- `tests/test_waveform.py`
- `tests/test_time_grid.py`
- `tests/test_error_metric_floor.py`
- `tests/test_dolfinx_validation_artifacts.py`
- `tests/test_prony.py`

## Validation Command

Lightweight P0-P2 tests:

```bash
python -m pytest -q tests/test_waveform.py tests/test_time_grid.py tests/test_error_metric_floor.py tests/test_dolfinx_validation_artifacts.py tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_solves_through_ramp_then_returns_observation_times tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_keeps_ramp_grid_when_observations_end_before_ramp tests/test_dolfinx_model_consistency.py::test_after_ramp_observation_schedule_uses_ramp_solver_t_min_before_later_observation_start
```

P3 material-interface tests:

```bash
python -m pytest -q tests/test_prony.py tests/test_ip_model.py tests/test_debye_fit.py
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
  --rho-air 1e6 \
  --rho-earth 100 \
  --layer-depths 350,650 \
  --layer-resistivities 100,100,100 \
  --empymod-srcpts 21
```

The run writes both the legacy combined plot and the new P2 validation artifacts.

## Current Accuracy Status

The latest known no-IP baseline still exceeds 5% over the full time range:

- `Ex max` is roughly tens of percent.
- `Hz/dBzdt` also exceeds 5% depending on recovery mode and time window.

This implementation round improves time-axis correctness and reporting/diagnostics. It does not resolve the known near-source source-transfer/MMR consistency problem.

## Known Limitations

- `diagnose_source_consistency` currently reports waveform-integral and endpoint-total checks without full FEM matrix residuals unless a source projection residual is provided.
- Average receivers and Faraday-integrated `Hz` recovery are not fully implemented in this round.
- P3 currently provides the material API and memory-update tests; DOLFINx total-field IP assembly still needs to be migrated to this API and verified against no-IP when `delta_sigma=0`.
- Primary-secondary solvers remain for P4+.
- Full no-IP/IP 5% acceptance is not yet achieved.

## Next Steps

1. Add real FEM source residual diagnostics using assembled gradient/divergence/curl operators.
2. Add `point`, `volume_average`, and `disk_average` receiver modes in DOLFINx.
3. Add Faraday-integrated magnetic recovery as an alternative to Biot-Savart `Hz`.
4. Continue P3 by wiring `PronyConductivity` into DOLFINx total-field IP assembly and adding solver-level `delta_sigma=0` no-IP equivalence tests.
5. Continue to P4: `PrimaryFieldProvider` abstraction.
