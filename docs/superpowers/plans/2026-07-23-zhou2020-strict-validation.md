# Zhou 2020 Grounded-Wire TEM-IP Strict Validation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Apply superpowers:test-driven-development for every code change and superpowers:verification-before-completion before any passing claim.

**Goal:** Replace Song as the formal TEM-IP benchmark with a traceable Zhou 2020 grounded-wire layered model, compare independent empymod and FEniCSx responses over the full polarization time window, perform numerical convergence checks, and deliver a rendered, evidence-backed Chinese Word validation report.

**Architecture:** Keep the existing solver and fail-closed SOTEM artifact machinery, but add a Zhou-specific parameter/provenance contract and a small strict-validation orchestration layer. The empymod path evaluates the exact Pelton complex resistivity independently; the FEniCSx path fits the same conductivity with Debye terms and never reads reference responses before comparison. Published run bundles are immutable and manifest-backed. The Word builder consumes only verified bundles and distinguishes journal-figure evidence from companion-disclosure parameters.

**Tech Stack:** Python 3.10/3.11, NumPy, SciPy, empymod, PyYAML, pytest, FEniCSx/DOLFINx 0.8, PETSc, MPI, Gmsh, matplotlib, python-docx, LibreOffice/Poppler rendering.

---

## Task 1: Create the Zhou benchmark and parameter-provenance contract

**Files:**

- Create: `benchmarks/sotem/zhou2020_grounded_wire.yaml`
- Create: `benchmarks/sotem/zhou2020_parameter_provenance.json`
- Modify: `tests/test_sotem_benchmark.py`
- Modify: `src/atem3d/sotem_benchmark.py`

**Step 1: Write failing tests**

Add tests asserting:

- source `(-500,0,0)` to `(500,0,0)`, `10 A`, ideal step-off;
- receiver `(0,1000,0)`;
- layers `100 ohm m / 500 m`, `10 ohm m / 20 m`, `200 ohm m halfspace`;
- polarization only in `500-520 m`, `rho0=10`, `m=0.1`, `tau=1`, `c=0.3`;
- `101` log-spaced times from `1e-4` to `3 s`;
- provenance has a source URL and evidence grade for every physical field;
- no field has evidence grade `X`;
- the Song case is tagged `qualitative_only` or excluded from the strict suite.

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_benchmark.py -k "zhou or provenance" -v
```

Expected: FAIL because the case and provenance contract do not exist.

**Step 2: Implement the minimum schema changes**

Extend `BenchmarkCase` only as needed to retain:

- validation role;
- literature identifiers;
- numerical surface offset policy;
- provenance-file linkage.

Do not weaken validation of existing Lei/Song YAML files. Load the provenance JSON independently and validate exact case-id/hash binding.

**Step 3: Run targeted and full benchmark tests**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_benchmark.py -v
```

**Step 4: Commit**

```powershell
git add benchmarks/sotem/zhou2020_grounded_wire.yaml benchmarks/sotem/zhou2020_parameter_provenance.json src/atem3d/sotem_benchmark.py tests/test_sotem_benchmark.py
git commit -m "feat: add traceable Zhou grounded-wire benchmark"
```

## Task 2: Add an exact Pelton empymod material reference

**Files:**

- Modify: `src/atem3d/empymod_compare.py`
- Modify: `src/atem3d/materials/cole_cole.py`
- Modify: `tests/test_empymod_compare.py`
- Modify: `tests/test_materials_cole_cole.py`

**Step 1: Write failing tests**

Test the exact Pelton relation:

`rho*(w)=rho0*[1-m*(1-1/(1+(iw tau)^c))]`

and verify:

- `sigma(0)=1/rho0`;
- `sigma(infinity)=1/[rho0(1-m)]`;
- only the second earth layer disperses;
- air, first layer and halfspace remain frequency-independent;
- empymod receives exact complex conductivity through `func_eta`;
- no Debye fit is used in the exact-reference function.

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_materials_cole_cole.py tests/test_empymod_compare.py -k "exact_pelton" -v
```

Expected: FAIL.

**Step 2: Implement exact Pelton conductivity**

Add a public helper such as:

```python
make_exact_pelton_resistivity_model(
    rho0_by_layer,
    chargeability_by_layer,
    tau_by_layer,
    c_by_layer,
)
```

The helper must calculate `sigma*(frequency)=1/rho*(frequency)` without calling the Debye fitter.

**Step 3: Verify against analytical limits**

Run the targeted tests and the full empymod/material test files.

**Step 4: Commit**

```powershell
git add src/atem3d/empymod_compare.py src/atem3d/materials/cole_cole.py tests/test_empymod_compare.py tests/test_materials_cole_cole.py
git commit -m "feat: add exact Pelton empymod reference"
```

## Task 3: Build the strict empymod reference and source-integration convergence runner

**Files:**

- Create: `src/atem3d/zhou2020_reference.py`
- Create: `src/atem3d/zhou2020_reference_cli.py`
- Create: `tests/test_zhou2020_reference.py`
- Modify: `pyproject.toml`

**Step 1: Write failing tests**

Cover:

- no-IP and IP variants;
- canonical columns `Ex_V_per_m`, `Hz_A_per_m`, `dBzdt_T_per_s`;
- signed finite data;
- `srcpts` sequence `3,5,9,17`;
- convergence computed with a 1%-of-reference-peak floor;
- adjacent 9-to-17 strong-response change gate `<=0.5%`;
- surface-offset sensitivity at `0,0.05,0.1,0.2 m`;
- metadata recording empymod version, filter options, model hash and source integration;
- atomic output and fail-closed status.

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_reference.py -v
```

Expected: FAIL.

**Step 2: Implement the runner**

The CLI accepts:

```text
--case
--provenance
--output-dir
--srcpts 3,5,9,17
--surface-offsets 0,0.05,0.1,0.2
```

Write:

- `empymod_noip.csv`;
- `empymod_ip.csv`;
- `empymod_srcpts_convergence.json`;
- `surface_offset_sensitivity.json`;
- `empymod_metadata.json`;
- `reference_manifest.json`.

**Step 3: Run tests and a real reference calculation**

Windows:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_reference.py -v
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m atem3d.zhou2020_reference_cli --case benchmarks/sotem/zhou2020_grounded_wire.yaml --provenance benchmarks/sotem/zhou2020_parameter_provenance.json --output-dir generated/validation/zhou2020_grounded_wire/reference-preflight
```

**Step 4: Inspect signed curves**

Confirm no NaN/Inf, units, sign, source-integral convergence, and whether `Ex` exhibits the expected IP separation and reversal scale.

**Step 5: Commit**

```powershell
git add src/atem3d/zhou2020_reference.py src/atem3d/zhou2020_reference_cli.py tests/test_zhou2020_reference.py pyproject.toml
git commit -m "feat: add Zhou empymod reference workflow"
```

## Task 4: Add strict comparison metrics and zero-crossing evidence

**Files:**

- Create: `src/atem3d/zhou2020_metrics.py`
- Create: `tests/test_zhou2020_metrics.py`

**Step 1: Write failing tests**

Test:

- full-window relative L2;
- robust point error with 1%-peak denominator floor;
- strong/weak response classification;
- pure IP difference `IP-noIP`;
- zero crossings interpolated in log-time;
- no deletion of zero-near samples;
- explicit `incomplete_time_window`;
- gates fixed at 5% total field, 10% IP increment, 10% first zero-crossing time;
- failed component/time retention.

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_metrics.py -v
```

Expected: FAIL.

**Step 2: Implement metrics**

Do not alter legacy Song metrics. Use a dedicated strict result schema:

`atem3d.zhou2020.strict-comparison/v1`.

**Step 3: Run tests**

Run targeted tests and `tests/test_sotem_metrics.py` to prove no regression.

**Step 4: Commit**

```powershell
git add src/atem3d/zhou2020_metrics.py tests/test_zhou2020_metrics.py
git commit -m "feat: add strict Zhou comparison metrics"
```

## Task 5: Connect the Zhou case to the existing FEniCSx pipeline

**Files:**

- Modify: `dolfinx/run_sotem_benchmark.py`
- Modify: `dolfinx/sotem_pipeline.py`
- Create: `benchmarks/sotem/run_zhou2020_fenicsx.sh`
- Modify: `tests/test_run_sotem_benchmark.py`
- Modify: `tests/test_dolfinx_model_consistency.py`
- Modify: `tests/test_dolfinx_mesh_refinement_config.py`

**Step 1: Write failing adapter tests**

Assert:

- all three layers map correctly in `z_up`;
- the `500 m` and `520 m` interfaces are explicit mesh/material boundaries;
- the second layer alone receives `rho0=10,m=0.1,tau=1,c=0.3`;
- no-IP keeps the same DC resistivity and geometry;
- `sigma0=0.1 S/m`, `sigma_inf=1/9 S/m`;
- output times cover `1e-4-3 s`;
- `S0/S1/S2` resolve the 20 m layer with nominal `10/5/2.5 m`;
- canonical surface and numerical offset are both recorded;
- `--check-env-only` reaches the real pipeline without solving.

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_run_sotem_benchmark.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_mesh_refinement_config.py -k "zhou" -v
```

Expected: FAIL.

**Step 2: Implement only the required adapter changes**

Reuse existing Nedelec, DC, magnetic recovery, Debye and receiver code. Do not create a second solver. Add the minimum mesh-interface and time-window configuration needed for the Zhou case.

**Step 3: Verify WSL environment**

```powershell
wsl -d Ubuntu -- bash -lc "source /home/paidaxin/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx && cd '<worktree-wsl-path>' && python dolfinx/run_sotem_benchmark.py --case benchmarks/sotem/zhou2020_grounded_wire.yaml --variant noip --level S0T0B0 --workdir /tmp/zhou-contract --check-env-only --no-install"
```

Do not use `mpiexec ... python -`.

**Step 4: Run targeted Windows and WSL tests**

**Step 5: Commit**

```powershell
git add dolfinx/run_sotem_benchmark.py dolfinx/sotem_pipeline.py benchmarks/sotem/run_zhou2020_fenicsx.sh tests/test_run_sotem_benchmark.py tests/test_dolfinx_model_consistency.py tests/test_dolfinx_mesh_refinement_config.py
git commit -m "feat: connect Zhou benchmark to FEniCSx"
```

## Task 6: Add fail-closed run orchestration and manifest verification

**Files:**

- Create: `src/atem3d/zhou2020_validation_cli.py`
- Create: `tests/test_zhou2020_validation_cli.py`
- Modify: `pyproject.toml`

**Step 1: Write failing orchestration tests**

Cover:

- unique run-id;
- immutable case/provenance snapshots;
- safe relative evidence paths;
- atomic stage publication;
- reference must complete before FEniCSx comparison;
- no-IP must pass before IP production run;
- partial window status cannot become validated;
- config/result hashes checked before report generation;
- failed evidence preserved;
- final status only becomes `ip_internally_validated` when every required gate exists and passes.

Run:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_zhou2020_validation_cli.py -v
```

Expected: FAIL.

**Step 2: Implement subcommands**

Use:

```text
prepare
reference
register-fenicsx
compare
finalize
verify
```

Reuse safe manifest helpers from `sotem_validation_cli.py` where possible; do not duplicate unsafe file-writing logic.

**Step 3: Run tests and create the production run directory**

**Step 4: Commit**

```powershell
git add src/atem3d/zhou2020_validation_cli.py tests/test_zhou2020_validation_cli.py pyproject.toml
git commit -m "feat: orchestrate strict Zhou validation"
```

## Task 7: Run staged no-IP and IP FEniCSx validation

**Files produced outside Git:**

- `generated/validation/zhou2020_grounded_wire/<run-id>/...`

**Step 1: Create run and publish empymod reference**

Run `prepare` then `reference`, record the actual run-id.

**Step 2: Run S0T0B0 no-IP smoke**

Use five diagnostic times first:

`1e-4,1e-3,1e-2,1e-1,1 s`.

Preserve failure evidence. Do not tune amplitudes, signs or thresholds.

**Step 3: Diagnose and fix only proven no-IP defects**

Use this order:

1. coordinates and source endpoints;
2. DC initial field;
3. finite-line source projection;
4. receiver location and components;
5. `Hz/Bz/dBzdt` units and signs;
6. thin-layer material tags;
7. time stepping;
8. boundaries;
9. solver tolerance.

For any code defect, add a failing regression test before the fix and commit the fix separately.

**Step 4: Run S1T1B1 no-IP full window**

Register results and compare against empymod.

**Step 5: Enable IP only after no-IP gate**

Run the Debye material gate, DC steady-state memory test, and `m->0` regression before the IP smoke.

**Step 6: Run IP smoke and full window**

Verify signed `Ex`, `Hz`, `dBzdt`, `IP-noIP`, and zero crossing.

**Step 7: Record actual status**

If a gate fails, preserve it as `failed_with_reproducible_evidence`; do not continue to expensive refinement until the failure is understood.

## Task 8: Run convergence matrix

**Files produced outside Git:**

- convergence artifacts in the same run-id.

**Step 1: Spatial refinement**

Run S0/S1/S2 with fixed T1/B1. Confirm thin-layer resolution and strong-response changes.

**Step 2: Temporal refinement**

Run T0/T1/T2 with fixed accepted spatial/boundary level.

**Step 3: Boundary and air sensitivity**

Run B0/B1/B2 and the specified air/surface-offset sensitivity checks.

**Step 4: Solver tolerance**

Re-run one representative no-IP and IP model at a stricter linear tolerance.

**Step 5: Final combined model**

Re-run the selected fine configuration, not merely infer it from one-factor sweeps.

**Step 6: Publish `convergence.json`**

Include pass/fail per quantity and actual resource statistics.

## Task 9: Acquire and register the literature response figure

**Files:**

- Create: `assets/reports/zhou2020/README.md`
- Create: `assets/reports/zhou2020/source_manifest.json`
- Add legally obtained page/figure renderings under `assets/reports/zhou2020/`
- Create: `data/paper_digitized/zhou2020_ex_hz.csv`
- Create: `tests/test_zhou2020_paper_evidence.py`

**Step 1: Register source before digitization**

Record DOI, figure number, page, URL, acquisition date, image hash, axis scale, labels and expected uncertainty.

**Step 2: Digitize only the necessary curves**

Digitize signed IP/no-IP `Ex` and `Hz` if the figure resolution permits. Never derive `dBzdt` by differentiating a digitized magnetic curve.

**Step 3: Write validation tests**

Require:

- source manifest and image hashes;
- monotonically increasing positive times;
- signed values;
- explicit curve labels;
- no claim of author-provided tabulated data;
- digitization uncertainty field.

**Step 4: Commit only reproducible, license-compliant evidence**

If the journal figure cannot legally be stored, keep the manifest, citation and derived digitized points, and let the report builder reference rather than embed the original.

## Task 10: Build plots and the Chinese Word report

**Files:**

- Create: `scripts/reports/build_zhou2020_strict_validation_report.py`
- Create: `scripts/reports/verify_zhou2020_strict_validation_report.py`
- Create: `tests/test_build_zhou2020_strict_validation_report.py`
- Produce: `output/doc/Zhou2020接地线源TEM-IP_FEniCSx_empymod严格验证报告.docx`

**Step 1: Load the document skill and runtime dependencies**

Read the complete documents skill references, call `codex_app.load_workspace_dependencies`, and use its supported `python-docx` and rendering stack.

**Step 2: Write failing report tests**

Require:

- title and source citations;
- explicit statement that Song is not the formal benchmark;
- complete parameter table;
- exact Pelton convention and `sigma0/sigma_inf`;
- signed no-IP/IP plots;
- pure IP increment plots;
- zero-crossing table;
- source integration, mesh, time and boundary convergence;
- pass/fail matrix tied to machine-readable summary;
- limitations and internal-validation wording;
- hashes and reproduction commands;
- no placeholders;
- report status matching the manifest.

**Step 3: Implement the report builder**

Generate all figures from published CSV/JSON evidence. Do not hard-code computed results in prose.

**Step 4: Run report tests**

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_build_zhou2020_strict_validation_report.py -v
```

**Step 5: Build the DOCX**

Use a run-id argument and refuse unverified evidence.

**Step 6: Verify structure and values**

Run the dedicated verifier and compare extracted DOCX text against `comparison_summary.json`.

**Step 7: Commit report code and final report**

```powershell
git add scripts/reports/build_zhou2020_strict_validation_report.py scripts/reports/verify_zhou2020_strict_validation_report.py tests/test_build_zhou2020_strict_validation_report.py output/doc/Zhou2020接地线源TEM-IP_FEniCSx_empymod严格验证报告.docx
git commit -m "docs: add strict Zhou TEM-IP validation report"
```

## Task 11: Render and visually inspect every report page

**Files produced for QA:**

- `tmp/docs/zhou2020_validation/rendered/page-*.png`
- optional PDF in the same QA directory.

**Step 1: Render the final DOCX**

Use the bundled `render_docx.py` workflow.

**Step 2: Inspect every PNG page**

Check every page for:

- clipped figures/tables;
- unreadable axes;
- Chinese font substitution;
- broken formulas;
- orphan headings;
- blank pages;
- incorrect captions;
- pass/fail colors that do not match text.

**Step 3: Fix and re-render**

Repeat until every page is acceptable. Do not verify only the first or last page.

**Step 4: Record visual QA**

Save a page-count and per-page inspection checklist in the report verification output.

## Task 12: Final completion audit

**Step 1: Run the relevant test suite**

Windows:

```powershell
D:\APP\ANACONDA\envs\Codex_simpeg_ATEM3D\python.exe -m pytest tests/test_sotem_benchmark.py tests/test_empymod_compare.py tests/test_materials_cole_cole.py tests/test_zhou2020_reference.py tests/test_zhou2020_metrics.py tests/test_zhou2020_validation_cli.py tests/test_run_sotem_benchmark.py tests/test_dolfinx_model_consistency.py tests/test_build_zhou2020_strict_validation_report.py -v
```

WSL/FEniCSx:

```powershell
wsl -d Ubuntu -- bash -lc "source /home/paidaxin/miniconda3/etc/profile.d/conda.sh && conda activate fenicsx && cd '<worktree-wsl-path>' && python -m pytest <relevant-dolfinx-tests> -v"
```

**Step 2: Verify production artifacts**

Freshly run:

- strict validation CLI `verify`;
- report verifier;
- file hashes;
- DOCX render;
- Git status.

**Step 3: Audit every design requirement**

Build a requirement-to-evidence checklist covering:

- parameter provenance;
- exact empymod reference;
- FEniCSx no-IP/IP full window;
- IP increment;
- zero crossing;
- convergence;
- literature response figure;
- Word;
- page-by-page visual QA.

Do not mark the goal complete if any evidence is missing, partial, indirect, or stale.
