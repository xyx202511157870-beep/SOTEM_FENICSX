# Flow 5 / G3D-0 — three-dimensional resource preflight

**Decision: `BLOCKED_BY_SOFTWARE_OR_RESOURCES`.**  
**G3D-0: BLOCKED.**  
**3-D was not started.** This document sizes D3-A / D3-B. It is not a G3D-1 result.

xin authorized Flow 5 only. Flow 6 full 3-D tests are not authorized. PR #10 stays draft.

Official artifacts referenced here are frozen and were not rewritten:

- `LAYERED_GATE_PASSED.json` (`L2_PASS`, `3D_AUTHORIZED_PENDING_PREFLIGHT`)
- `selected_method.json` / `selected_template_by_K.json` / `spectral_selected_template_by_K.json`
- `FLOW1_STATUS.json` … `FLOW4_STATUS.json`

---

## 1. Decision and G3D-0

G3D-0 asks whether a fair 3-D campaign (independent D3-A / D3-B, audited REF, frozen B2 and P-R, numerical floor) can run on the available host.

| Check | Result | Why |
|---|---|---|
| Real FEniCSx / PETSc / Gmsh on this VM | FAIL | `import dolfinx` binds the repo folder `/workspace/dolfinx` (`__file__ is None`). `petsc4py`, `gmsh`, `mpi4py`, `ufl`, `basix` are missing. No conda `fenicsx` prefix. Flow 0 `dolfinx=present` was a **namespace collision**, not a solver install. |
| Memory for a fair M1 Cole–Cole REF | FAIL | Pipeline estimator at `dolfinx/sotem_pipeline.py:1473–1494` gives **64.4 GB** for D3-A M1 (~3.5e5 tets, K=20). This VM has **15.6 GiB RAM, 0 swap, nproc=1**. |
| Wall-clock for core 10 runs (5+5) | FAIL | Core estimate **17.7–41 h**; official cap **≤10 h total / ≤5 h per family**. Unsourced hour model — see §5. |
| Reusable audited 3-D REF | FAIL | Search documented in `3D_REFERENCE_PLAN.json`. No artifact meets the official floor. |
| Fair cheap-mesh substitute | Forbidden | A 1.65e5-cell P2 mesh that fits 32 GB cannot certify p95 ≤ 1.805e-4 (needs an M2 pair). |
| Fairness (same mesh/time/window/channels for REF, B2, P-R) | Cannot be executed here | Software + memory + hours all fail. |

**Independently sufficient blockers:** any one of (no FEniCSx, 64 GB vs 16 GB, 18–41 h vs 10 h) is enough to block. Combined they are decisive.

**Not `need xin decision`.** The facts are measured. xin already authorized preflight, not a 3-D campaign.

---

## 2. Environment probe

Written at Flow 5 start from `/proc`, `lscpu`, `nproc`, and Python imports. Full dump: `ENV_PROBE.json`.

| Item | Value |
|---|---|
| Host | Cursor cloud VM, Linux 6.12.94+ |
| Advertised CPUs | 4 (`lscpu`) |
| Usable processes | **1** (`nproc`; cgroup quota) |
| RAM | 16777216 kB ≈ **15.6 GiB** |
| Swap | **0** |
| Python | 3.12.3 |
| empymod | 2.6.0 (works; used only for the spectral pole probe) |
| Real `dolfinx` package | **absent** |
| `petsc4py` / `gmsh` / `mpi4py` / `ufl` / `basix` | **absent** |
| conda `fenicsx` | **absent** |
| `import dolfinx` | binds **repo folder** `/workspace/dolfinx` |

No FEniCSx install was attempted. A smoke that cannot import the solver is not a sizing probe.

---

## 3. Official numerical error floor

Layered L2 numbers (unchanged; from `LAYERED_GATE_PASSED.json` / `FLOW4_STATUS.json`):

| Quantity | Value |
|---|---|
| Official P-R / B2 median | `0.382566169465141` |
| Official 95% CI | `[0.163401575828889, 0.7770330921368851]` |
| Official win_rate | `0.9` |
| Official median `e_pr` | `4.0380472011408475e-4` |
| Official median `e_b2` | `1.0221694364605988e-3` |
| Official median gap `e_b2 − e_pr` | `5.416473400e-4` |
| One-third of that gap | `1.805491133e-4` |
| 0.3% absolute | `3.0e-3` |

**Official 3-D numerical floor = min(3.0e-3, 1.805491133e-4) = `1.805491133e-4`.**

This floor applies to the p95 of mesh / time / outer-boundary / solver / magnetic-recovery / disk-quadrature residuals on the six channels plus the IP increment. The expected method gap used for sizing is the official layered median gap `5.416e-4`. A 3-D gap is unknown until Flow 6; the layered gap is the only audited scale available for G3D-0.

---

## 4. Memory model (sourced)

The only in-repo memory model is `estimate_resource_budget` in `dolfinx/sotem_pipeline.py`.

```
estimated_gb = 4.0 * (n_cells * 2.85e-5 + n_nodes * 1.5e-6) * (1 + 0.03 * n_terms)
              if cole-cole / Debye, else without the (1 + 0.03 K) factor
usable_gb    = 0.95 * memory_limit_gb
```

Empirical check (recomputed in this Flow 5 session, not taken from a stale table):

| n_cells | n_nodes | no-IP GB | K=10 GB | K=20 GB |
|---|---:|---:|---:|---:|
| 1.044e6 | 164973 | 120.00 | 156.00 | 192.00 |

That matches the comment at `sotem_pipeline.py:1488` (`~1.04e6 cells → 120 GB`).

P2 Nédélec DOF uses the planning factor **6.4 × n_cells** from `3D_REFERENCE_PLAN.json` (`dof_model.dof_per_cell`; unsourced — exact N1curl degree-2 count is 6 dofs/tet plus traces). Node count for a graded tet mesh is taken as `n_nodes ≈ 0.18 × n_cells` when a mesh file is absent (same ratio as the 1.044e6 / 164973 example). Memory depends on cells+nodes, not on the 6.4 factor.

### 4.1 Fit tables used for D3-A / D3-B

| Mesh | n_cells | n_nodes (est.) | P2 DOF (6.4×) | no-IP GB | K=10 GB | K=20 GB | K=24 GB | K=28 GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D3-A M1 | 3.50e5 | 6.30e4 | 2.2e6 | 40.3 | 52.3 | **64.4** | 69.3 | 74.1 |
| D3-A M2 | 1.00e6 | 1.80e5 | 6.4e6 | 115.1 | 149.6 | **184.1** | 198.0 | 211.8 |
| D3-B M1 | 4.00e5 | 7.20e4 | 2.6e6 | 46.0 | 59.8 | **73.6** | 79.1 | 84.7 |
| D3-B M2 | 1.15e6 | 2.07e5 | 7.4e6 | 132.3 | 172.0 | **211.8** | 227.6 | 243.5 |
| 32 GB workstation max at P2 K=20 | 1.651e5 | 2.97e4 | 1.06e6 | 19.0 | 24.7 | 30.4 | 32.7 | 35.0 |

D3-A M1 K=20 already exceeds this VM by 4× and a 32 GB workstation by 2×. D3-A M2 K=20 is a 184 GB machine. The 1.65e5-cell “cheap mesh” is recorded only to show why it is rejected: it is 2.1× coarser than M1 and cannot carry the official floor.

---

## 5. Hour model (unsourced — do not treat as measured)

There is **no audited 3-D wall-time** in this repository. Existing `generated/sotem3d_*` and `paper_algorithm/sotem3d_*` folders have no `manifest.json` / `timing.json`. Therefore every hour number below is a **planning model**, not a measurement.

### 5.1 Model

```
T_hours ≈ 1.35 * (n_cells / 3.5e5) * (n_steps / 130) * (1 + 0.035 * K) * (1 + 0.15 * [disk])
          * (4 / max(nproc_effective, 1))
```

Constants:

- `1.35 h` = assumed wall time of one D3-A M1, K=10, 130-step, point-receiver run on **4 ranks**.
- `n_steps = 130` = representative of the official 1e-5–1e-2 window (internal traces show ~124–134 accepted steps on W0/W1/W2).
- `0.035 * K` = Debye-term linearization consistent with the memory `0.03 * K` slope, slightly steeper because each term is a field.
- `0.15` = disk / AverageReceiver quadrature (36-point) overhead relative to a point receiver.
- `4 / nproc` = this VM is **nproc=1**, so the same run is modeled **4× slower** here than on a 4-rank workstation.

If the 1.35 h constant is wrong by 2×, every hour number in this file moves by 2×. The block does **not** depend on that constant: even if the constant were 0.4 h, D3-A M1 K=20 on 1 rank is still ~2.1 h and the **memory** block remains.

### 5.2 Per-run hours (authoritative envelope = `3D_REFERENCE_PLAN.json`)

These are the numbers the G3D-0 decision uses. They assume a 4-rank ≥128 GB host. This VM (1 rank, 16 GB) cannot run them.

| Run | Mesh | K | est. DOF | GB | Hours lo–hi |
|---|---|---:|---:|---:|---|
| A-noIP | M1 | — | 2.2e6 | 40.3 | 1.5–3.6 |
| A-REF-K20 | M1 | 20 | 2.2e6 | 64.4 | 1.7–4.0 |
| A-REF-K24 | M1 | 24 | 2.2e6 | 69.3 | 1.7–4.0 |
| A-B2-K10 | M1 | 10 | 2.2e6 | 52.4 | 1.6–3.8 |
| A-PR-K10 | M1 | 10 | 2.2e6 | 52.4 | 1.6–3.8 |
| A-REF-K20-M2 | M2 | 20 | 6.4e6 | 184 | 4.5–11 |
| B-noIP | M1 | — | 2.6e6 | 46.0 | 1.8–4.2 |
| B-REF-K20 | M1 | 20 | 2.6e6 | 74.0 | 2.0–4.5 |
| B-REF-K24 | M1 | 24 | 2.6e6 | 79.0 | 2.0–4.5 |
| B-B2-K10 | M1 | 10 | 2.6e6 | 60.0 | 1.9–4.3 |
| B-PR-K10 | M1 | 10 | 2.6e6 | 60.0 | 1.9–4.3 |
| B-REF-K20-M2 | M2 | 20 | 7.4e6 | 212 | 5.0–13 |

### 5.3 Campaign totals vs official budget

Official cap: **≤5 h / family, ≤10 h total**.

| Campaign | Runs | Hours (envelope) | Peak GB | Fits 10 h? | Fits 16 GB? |
|---|---:|---|---:|:---:|:---:|
| D3-A core (no-IP + REF20 + REF24 + B2 + P-R) | 5 | 8.1–19.2 | 69 | no | no |
| D3-A full floor | 10 | 24–56 | 184 | no | no |
| D3-B core | 5 | 9.6–21.8 | 79 | no | no |
| D3-B full floor | 12 | 32–77 | 212 | no | no |
| Combined core | **10** | **17.7–41** | 79 | no | no |
| Combined full floor | **22** | **56–133** | 212 | no | no |

The block does **not** depend on the 1.35 h constant: memory alone (64.4 GB vs 15.6 GiB) and the absence of FEniCSx are independently sufficient.

**REF reuse: no.** Adding REF to every family is mandatory. Minimum fair count is **5 runs / family** if K_ref=20 converges against K=24, else 6.

---

## 6. Alternatives considered and rejected

| Alternative | Why rejected |
|---|---|
| 1.65e5-cell P2 mesh on a 32 GB host | Fits memory. Cannot certify p95 ≤ 1.805e-4 (no M2 pair). Forbidden by fairness. |
| Skip M2 / skip K=24 | Leaves the official floor and K_ref rule untested. |
| Reuse `sotem3d_line_source` / `paper_algorithm/sotem3d_*` | No `manifest.json`, no `config_hash`, no `git_commit`, no six-channel + IP-increment residuals, no frozen K=10 templates. See `3D_REFERENCE_PLAN.json`. |
| Claim Flow 0 `dolfinx=present` as a solver | Namespace collision with `/workspace/dolfinx`. |
| Install FEniCSx on this VM and “try a smoke” | Designer + xin: no solver-core work; no install-to-try; a 16 GB / 1-rank host still cannot run M1. |
| Start Flow 6 on a coarser mesh “just to see” | xin authorized Flow 5 only. A cheap 3-D number would be claimed as evidence. Not done. |

---

## 7. Case families (summary)

Full geometry, sources, windows, channels, and scored-series lists: `3D_CASE_CONTRACT.json`.

Independence from the layered registry (`case_registry.csv`, 40 rows): no row satisfies `|Δlog10 τ| < 0.05` and `|Δm| < 0.02` and `|Δc| < 0.02` against D3-A (`τ=2e-3, m=0.22, c=0.55`) or D3-B (`τ=8e-3, m=0.30, c=0.45`). Nearest rows are TR02 and TR08. Result **PASS** — families were not used in layered selection.

| Family | Body | Source | Windows | Receiver | Scored series | M1 cells / DOF | M2 cells / DOF |
|---|---|---|---|---|---:|---|---|
| D3-A | Canonical prism 200×300×60 m, top 120 m, CC `ρ0=40, m=0.22, τ=2e-3, c=0.55` | Grounded wire A(−420,−150)→B(380,90), L=835.22 m, az=16.70° | W1, W2 | Point (120, 520, −0.1) | 12 | 3.5e5 / 2.2e6 | 1.0e6 / 6.4e6 |
| D3-B | OCC union: 35° dipping slab + vertical conduit; overburden 30 Ω·m / basement 150 Ω·m; CC `ρ0=15, m=0.30, τ=8e-3, c=0.45` | Grounded wire A(−520,−80)→B(260,260), L=850.88 m, az=23.55° | W3 (`t_us=[-40,-25,-10,0]`) | Point + disk r=1.0 at (40, 610, −0.1); tilted normal `[0.350048, -0.200028, 0.915126]` | 14 | 4.0e5 / 2.6e6 | 1.15e6 / 7.4e6 |

Asymmetric finite grounded sources. Six channels (Hx, Hy, Hz, dBx/dt, dBy/dt, dBz/dt) plus IP increment. Frame: right-handed, z up, ground z=0, numerical surface offset 0.1 m. Window 1e-5–1e-2 s, 31 log samples.

---

## 8. K_ref plan

Candidates: **20, 24, 28**. Rule (also in `3D_REFERENCE_PLAN.json`):

1. Build a spectral Cole–Cole fit whose relative residual on `σ*(ω)` is ≤ 1e-4 on the official band. The naive log-window probe in `REF_SPECTRAL_PROBE.json` **does not** meet 1e-4 (D3-A ~1.77e-3, D3-B ~1.14e-2). Flow 6 must place poles; it must not ingest the naive grid as REF.
2. On the 3-D mesh, compare REF_K vs REF_{K+4}. Accept the smaller K if the six-channel + IP-increment p95 ≤ 1.805e-4.
3. Never exceed K=28. If K=28 still misses, the family is blocked.
4. Label: **“K_ref Debye reference”**, never “exact Cole–Cole”.

Frozen comparison methods (do not refit):

- P-R K=10: `K10_cc_span4.0_shift-0.5_dens1.00`
- B2 K=10: `K10_tw_span6.0_shift+0.5_dens1.25`

`--cole-n-terms` in the current CLI refits Prony and **cannot** ingest those templates. That is a Flow 6 wiring gap, not a solver-core bug.

---

## 9. Feature gaps (Flow 6 wiring, not solver-core bugs)

Recorded so Flow 6 does not discover them after a 128 GB host appears. **`dolfinx/sotem_pipeline.py` was not edited.**

| Gap | Current code | Needed for D3-A / D3-B |
|---|---|---|
| Polarization support | Layered 1-D only | Body-local Cole–Cole / Debye on the prism / OCC union |
| Frozen templates | `--cole-n-terms` refits Prony | Ingest `selected_template_by_K.json` for B2 and P-R |
| W3 | W0 / W1 / W2 CLI | Tabulated `t_us=[-40,-25,-10,0]`, scales `[1, 0.80, 0.25, 0]` |
| Disk receiver | 5-point cross | 36-point `AverageReceiver` (paper_algorithm) |
| Tilted normal | Vertical assumed | Unit normal `[0.350048, -0.200028, 0.915126]` |

---

## 10. What this Flow 5 did and did not do

**Did:**

- Defined D3-A and D3-B in `3D_CASE_CONTRACT.json`.
- Searched the repo for reusable REF; recorded the negative result in `3D_REFERENCE_PLAN.json`.
- Probed this VM (`ENV_PROBE.json`) and a numpy-only spectral residual (`REF_SPECTRAL_PROBE.json`). The spectral probe is **not** a 3-D forward and is **not** G3D-1.
- Applied the official floor `1.805e-4` and the pipeline memory estimator.
- Wrote this budget, `G3D0_STATUS.json`, and `FLOW5_STATUS.json`.
- Updated root `LAYERED_DECISION.md` with the preflight decision. L0 / L1 / L2 numbers are unchanged.

**Did not:**

- Run any 3-D FEniCSx forward.
- Generate a Gmsh mesh.
- Invoke `run_sotem_benchmark.py`, `run_ip_debye_sweep.py`, or `run_algorithm_paper.sh`.
- Modify `dolfinx/sotem_pipeline.py` or any solver-core file.
- Modify `protocol.md`, `LAYERED_GATE_PASSED.json`, or `FLOW1`–`FLOW4_STATUS.json`.
- Claim G3D-1.
- Start Flow 6.

---

## 11. Next authorized step

None on this VM.

A later Flow 6 requires, at minimum:

1. A host with **real FEniCSx + PETSc + Gmsh**, **≥128 GB RAM** (256 GB if M2 / floor pairs run), and **≥4 MPI ranks**.
2. Flow 6 wiring for the five gaps in §9.
3. A new xin authorization for Flow 6. This preflight is not that authorization.

Until those exist, the official status stays:

`L0 passed. L1 is frozen. L2 passed (L2_PASS; layered outcome 3D_AUTHORIZED_PENDING_PREFLIGHT). Flow 5 G3D-0 preflight: BLOCKED. FINAL STATUS=BLOCKED_BY_SOFTWARE_OR_RESOURCES. 3-D was not started.`
