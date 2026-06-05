# DOLFINx Sponge Boundary Validation Notes

## Current status

The DOLFINx direct-time E-form now supports an optional transient sponge
conductivity shell.  The sponge is disabled by default.  When enabled, it is
applied to the transient conductivity matrices while the DC/on-time initial
field uses the physical conductivity unless `--sponge-apply-to-initial` is set.

This is not yet a final acceptance result.  The target remains:

- non-polarizable response error below 5% against empymod/analytic reference;
- polarizable response error below 5% against the corresponding reference;
- components must include the magnetic response, especially `dBz/dt`.

## Verified code checks

WSL environment:

- Python: `/home/paidaxin/miniconda3/envs/fenicsx/bin/python`
- DOLFINx: `0.8.0`
- empymod: `2.5.4`

Command:

```bash
python -m pytest \
  tests/test_dolfinx_sponge_config.py \
  tests/test_dolfinx_model_consistency.py \
  tests/test_dolfinx_runtime_report.py \
  -q
```

Result:

```text
60 passed
```

## Mesh and memory finding

The memory-safe candidate mesh with receiver-local `2.5 m` refinement is about
one million tetrahedra, which is feasible but slow on the 32 GB machine.

Mesh-only artifact:

```text
dolfinx/sponge_uniform200_receiver2p5_t1e3_meshonly/verification_mesh.msh
nodes: 169104
tetrahedra loaded by DOLFINx: 1050808
Nedelec dofs in solve: 1220301
```

This mesh used `air_height=5000 m` and `earth_depth=8000 m`.  It did not match
the previously passing early-time behavior as well as the older
`air_height=3000 m`, `earth_depth=6000 m` mesh.

## Non-polarizable uniform 200 ohm m checks

### Passing early window on the older depth-6000 mesh

Artifact:

```text
dolfinx/uniform200_receiver2p5_8km_depth6000_current_early
```

Settings:

```text
x_extent/y_extent: 8000 m
air_height: 3000 m
earth_depth: 6000 m
receiver_mesh_size: 2.5 m
time window: 1e-5 to 1.1e-5 s
sponge: disabled
```

Result:

```text
Ex max:      3.031191e-02
dBz/dt max: 1.627854e-02
Eh max:      3.870523e-02
```

This confirms the current code did not break the previously passing early
receiver-refined benchmark.

### Failing 1e-4 window without sponge

Artifact:

```text
dolfinx/uniform200_receiver2p5_depth6000_t1e4_nosponge
```

Runtime:

```text
total: 2164.767 s
forward solve: 2157.682 s
```

Result:

```text
Ex max:      4.958425e-02
dBz/dt max: 8.085737e-02
Eh max:      5.459400e-02
```

Interpretation:

- `Ex` is just below 5%.
- `Eh` is slightly above 5%, mostly because weak `Ey` absolute error adds to
  the horizontal vector norm.
- `dBz/dt` grows to about 8% by `1e-4 s`; this is now the main blocker for the
  non-polarizable short window.

### Failing 1e-4 window with strong sponge

Artifact:

```text
dolfinx/sponge_uniform200_receiver2p5_depth6000_t1e4
```

Settings:

```text
sponge_strength: 0.01 S/m
sponge_thickness: 1500 m
sponge_sides: x_min,x_max,y_min,y_max,z_min,z_max
apply_to_initial: false
```

Result:

```text
Ex max:      6.440951e-02
dBz/dt max: 8.070077e-02
Eh max:      6.947375e-02
```

Interpretation:

- This sponge setting is too strong/too close for this early-to-mid window.
- It does not solve the `dBz/dt` issue and worsens electric-field metrics.
- Future sponge sweeps should use weaker/farther shells and should not be
  judged on this single aggressive setting.

### Partial smaller time-growth check

Artifact:

```text
dolfinx/uniform200_receiver2p5_depth6000_t1e4_nosponge_tg105
```

The run stopped before completion after writing `forward_partial.npz` with 28
output times through `3.733456e-5 s`.  Postprocessing this partial data gave:

```text
Ex max:      4.923782e-02
dBz/dt max: 1.833872e-02
Eh max:      5.529783e-02
```

Interpretation:

- Decreasing `time_growth` to `1.05` did not improve `Eh` in the completed
  partial window.
- The late `dBz/dt` blocker at `1e-4 s` remains unresolved because this run did
  not reach that time.

## Failed analytic initial-field probe

Artifact:

```text
dolfinx/uniform200_receiver2p5_depth6000_t1e4_analyticdc
```

Using `--initial-dc-mode analytic_halfspace` on the one-million-cell mesh
exited after the first internal step without writing a report.  Subsequent WSL
diagnostics returned `Wsl/Service/E_UNEXPECTED`, so this path should not be
continued on the large mesh without first reproducing it on a smaller mesh.

Follow-up code fix:

- `_analytic_halfspace_dc_electric_field` now uses the normalized uniform earth
  resistivity from the active layer model instead of blindly using
  `config.rho_earth`.
- This matters when a uniform 200 ohm m audit is encoded as
  `--layer-depths 2000,2200 --layer-resistivities 200,200,200`; before the fix,
  analytic DC still used the default `rho_earth=100`.
- `validate_model_consistency` now rejects `--initial-dc-mode analytic_halfspace`
  for nonuniform layered models.
- CLI now also accepts `--rho-air` and `--rho-earth`, so uniform half-space runs
  no longer need fake layer definitions just to change the earth resistivity.

Verification after the fix was limited by WSL availability.  Windows Python
checks passed:

```powershell
python -m py_compile .\dolfinx\sotem_pipeline.py
python -m pytest tests/test_dolfinx_analytic_dc.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_sponge_config.py -q
```

Result:

```text
73 passed
```

WSL status after the large analytic-initial probe:

- `wsl.exe bash -lc "echo ok"` timed out repeatedly.
- `wsl.exe --shutdown` also timed out.
- Windows service discovery showed `WSLService` running, but restarting it did
  not restore command responsiveness during this session.
- A stronger cleanup attempt using `Stop-Process` and `taskkill` on `wsl.exe`
  / `wslservice.exe` also did not restore responsiveness.
- Therefore no further DOLFINx runtime smoke was performed after the analytic
  resistivity fix.  Continue with a small WSL smoke only after WSL responds to a
  trivial `echo ok` command.

Prepared smoke command for after WSL recovery:

```bash
./dolfinx/run_analyticdc_small_smoke.sh
```

This uses `--rho-earth 200` directly and should validate the corrected
`analytic_halfspace` initial field on a small mesh before returning to the
one-million-cell benchmark.

Long-run restart support added after the WSL failure:

- CLI flags:
  - `--checkpoint-forward`
  - `--resume-forward`
- Checkpoint file:
  - `workdir/forward_checkpoint.npz`
- Stored state:
  - completed internal step and previous internal time;
  - current Nedelec electric field state;
  - Debye memory arrays, if any;
  - completed receiver rows and solver log arrays;
  - Biot receiver magnetic state, if a Biot magnetic recovery mode is active.

This is intended for the one-million-cell validation runs, where a failure near
the end of the window should not force a full restart from `t=0`.

Windows-side verification after adding checkpoint support:

```powershell
$env:TMP='D:\Doctor\codex_app\simpeg自编时域电性源瞬变电磁法求解\.tmp'
$env:TEMP=$env:TMP
python -m pytest -p no:cacheprovider tests/test_dolfinx_partial_forward.py tests/test_dolfinx_analytic_dc.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_sponge_config.py -q
```

Result:

```text
75 passed
```

Segmented long-run support for the 32 GB workstation:

- CLI flag:
  - `--stop-after-outputs N`
- Behavior:
  - after `N` newly completed receiver output times, the E-form forward solver
    writes `forward_partial.npz`, saves `forward_checkpoint.npz`, and exits the
    time loop cleanly;
  - `--resume-forward --stop-after-outputs N` advances another `N` output
    times from the checkpoint instead of restarting from zero;
  - postprocessing now uses only `fem_result["times"]`, so partial runs compare
    empymod on the completed time samples only.

Windows-side verification after adding segmented stopping:

```powershell
$env:TMP='D:\Doctor\codex_app\simpeg自编时域电性源瞬变电磁法求解\.tmp'
$env:TEMP=$env:TMP
python -m pytest -p no:cacheprovider tests\test_dolfinx_partial_forward.py tests\test_dolfinx_analytic_dc.py tests\test_dolfinx_model_consistency.py tests\test_dolfinx_sponge_config.py -q
```

Result:

```text
77 passed
```

Current WSL status:

- `wsl.exe bash -lc "echo ok"` fails with the Windows message indicating that
  no Linux distribution is installed.
- `wsl.exe --list --verbose` fails with the same message.
- No DOLFINx runtime validation can be claimed until the Ubuntu/WSL
  distribution is restored.

32 GB memory preflight added:

- New CLI flags:
  - `--memory-limit-gb` (default `32`)
  - `--memory-safety-fraction` (default `0.95`)
- After Gmsh writes the `.msh`, the pipeline reads mesh statistics and estimates
  solver memory before loading the mesh into DOLFINx/PETSc.
- The preflight fails fast with `MemoryError` if the estimated solver memory is
  above the usable budget.  This is intentionally calibrated from observed
  1M-cell DOLFINx RSS and is meant to prevent a late PETSc/WSL crash on the
  32 GB workstation.
- `dolfinx/run_analyticdc_small_smoke.sh` now uses a script-relative project
  path instead of the previously garbled absolute WSL path, and passes the
  32 GB memory budget explicitly.

Windows-side verification after adding memory preflight:

```powershell
$env:TMP='D:\Doctor\codex_app\simpeg自编时域电性源瞬变电磁法求解\.tmp'
$env:TEMP=$env:TMP
python -m pytest -p no:cacheprovider tests\test_dolfinx_partial_forward.py tests\test_dolfinx_analytic_dc.py tests\test_dolfinx_model_consistency.py tests\test_dolfinx_sponge_config.py tests\test_dolfinx_mesh_refinement_config.py -q
python -m py_compile dolfinx\sotem_pipeline.py
```

Result:

```text
both commands passed
```

WSL recovery check:

- `wsl.exe bash -lc "echo ok && uname -a"` succeeded.
- Ubuntu WSL2 is running and the `fenicsx` conda environment contains DOLFINx,
  PETSc, gmsh, meshio, empymod, NumPy, SciPy, and matplotlib.
- `dolfinx/run_analyticdc_small_smoke.sh` completed successfully in WSL.  This
  validates that the WSL/DOLFINx path and the analytic-halfspace initial-field
  path are executable again, but the deliberately small mesh is not an
  acceptance-quality accuracy run.

Non-polarizable early-window rerun after WSL recovery:

- Workdir:
  `dolfinx/uniform200_receiver2p5_8km_depth6000_current_early`
- Reused the existing 1M-cell receiver-refined mesh.
- Runtime:
  - total `974.384 s`
  - forward solve `942.843 s`
- Result:
  - `Ex max=3.031191e-02`
  - `dBzdt max=1.627854e-02`
  - `Eh_vector max=3.870523e-02`
- This reconfirms the uniform 200 ohm m non-polarizable early window passes the
  5% physical metrics after WSL recovery.

Segmented `time_growth=1.05`, `t_max=1e-4` attempt:

- Workdir:
  `dolfinx/uniform200_receiver2p5_depth6000_t1e4_tg105_segmented`
- The process reached about `30 GB` RSS before writing the first
  `forward_partial.npz` or `forward_checkpoint.npz`.
- It was stopped manually to respect the 32 GB memory constraint.
- Root cause found: memory preflight was skipped when an existing mesh was
  reused.  The code now runs preflight for both newly generated and reused
  meshes, and the estimate is calibrated so the observed 1M-cell mesh is
  reported near 30 GB rather than the previous underestimated 19 GB.

Current-code no-IP and Cole-Cole acceptance reruns on the 100 m / 50 m model:

- Shared geometry:
  - source: `(-50, 0, -0.1)` to `(50, 0, -0.1)`, current `1 A`
  - receiver: `(500, 50, -0.1)`
  - source length `100 m`, parallel offset `50 m`
  - time window: `2.5e-6` to `1e-4 s`, `time_growth=1.25`
  - mesh: existing `395,093` loaded tetrahedra, `462,412` Nedelec dofs
  - memory preflight: estimated `11.822 GB` for no-IP and `15.369 GB` for
    Cole-Cole, both below the 32 GB workstation budget.

Non-polarizable rerun:

- Workdir:
  `dolfinx/runs/noip_offset50_afterramp_recv5_mean_src65_tmin25e7_t1e4`
- Runtime:
  - total `536.331 s`
  - forward solve `517.594 s`
- Physical acceptance metrics over the full window:
  - `Ex max=3.170741e-02`
  - `dBzdt max=2.682706e-02`
  - `Eh_vector max=3.568113e-02`
- empymod source-integration audit (`srcpts=65` vs `129`):
  - `Ex max=1.429022e-03`
  - `dBzdt max=4.697815e-05`
- This current-code run passes the physical `<5%` no-IP gate over the tested
  `2.5e-6` to `1e-4 s` window.

Cole-Cole polarizable rerun:

- Workdir:
  `dolfinx/runs/cole_offset50_afterramp_recv5_median_src65_tmin25e7_t1e4_debye_be`
- Debye fit:
  - Cole-Cole Debye relative L2 `2.607809e-03`
- Runtime:
  - total `583.767 s`
  - forward solve `561.338 s`
- Physical acceptance metrics over the full window:
  - `Ex max=3.186854e-02`
  - `dBzdt max=4.093153e-02`
  - `Eh_vector max=3.570948e-02`
- empymod source-integration audit (`srcpts=65` vs `129`):
  - `Ex max=7.377512e-08`
  - `dBzdt max=1.844770e-07`
- This current-code run passes the physical `<5%` Cole-Cole gate over the
  tested `2.5e-6` to `1e-4 s` window.

Important caveat:

- The weak transverse `Ey` component still has large standalone relative error
  because its empymod reference is near zero on this parallel-offset geometry.
- The acceptance gate therefore uses `Ex`, `dBzdt`, and horizontal electric
  vector norm `Eh_vector`, plus a weak-horizontal-component absolute gate:
  if a horizontal component has reference max below `0.1 * max(|Eh_ref|)`,
  its max absolute error is normalized by `max(|Eh_ref|)` and must be below
  `5%`.
- Current regenerated reports show this weak-component gate passes:
  - no-IP `Ey`: scaled absolute error max `1.799542e-02`
  - Cole-Cole `Ey`: scaled absolute error max `1.618693e-02`
- Literal per-component relative `Ey` percentages are still not used as the
  acceptance metric for this geometry, because that metric is dominated by the
  near-zero denominator rather than the physical response error.
- The automated acceptance test
  `tests/test_dolfinx_verified_acceptance.py` now points to the current no-IP
  and Cole-Cole rerun directories and checks both the physical response gate and
  the weak-horizontal-component gate.

Current comparison figures generated from these reruns:

- Geometry sketch:
  `dolfinx/figures/current_acceptance_model_geometry.png`
- Non-polarizable response and component-error figure:
  `dolfinx/figures/current_noip_three_component_empymod_error.png`
- Cole-Cole response and component-error figure:
  `dolfinx/figures/current_cole_three_component_empymod_error.png`

## Next technical steps

1. Decide the formal acceptance definition for the near-zero `Ey` component:
   physical vector-norm acceptance is currently below 5%, but literal
   per-component relative `Ey` is not meaningful on this geometry.
2. Generate the requested final comparison figures from the two current-code
   reruns: model sketch plus no-IP/Cole-Cole `Ex`, `Ey`, and `dBzdt` response
   and error panels.
3. After the 100 m / 50 m uniform halfspace acceptance package is finalized,
   return to the published layered model reproduction and repeat the same
   no-IP/Cole-Cole validation workflow.
