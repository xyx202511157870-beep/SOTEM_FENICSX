# 20 GB Layered Convergence Resource Contract Design

## Objective

Complete the approved paper-baseline layered convergence study on a workstation
whose final supported physical-memory limit is 20 GB. Preserve the locked
physical model, coordinate convention, local discretization, observation
window, and acceptance thresholds while replacing the infeasible 24 km domain
level with a predeclared 18 km level.

This document is a resource-constrained amendment to
`2026-07-12-paper-baseline-convergence-extension-design.md`. It supersedes only
that document's 24 km large-domain definition, 24 GB per-run limit, and
concurrency allowance. Every other scientific requirement remains in force.

## Evidence For The Amendment

The 24 km source-only mesh was generated before any 24 km field response was
computed. Its measured mesh statistics were 137,033 nodes, 851,996 retained
cell blocks, and 824,246 tetrahedra. The registered estimator reported 24.49 GB,
which exceeds both the former 24 GB guard and the newly confirmed 20 GB total
hardware limit. The 24 km forward solve was therefore never started.

The completed 12 km mesh estimate is 5.5844265 GB. A power-law interpolation
between the measured 12 km and 24 km mesh estimates predicts approximately
13.26 GB for an 18 km domain. This interpolation is planning evidence only. The
generated 18 km mesh must pass the exact estimator before any forward solve.

The domain replacement is selected from memory evidence alone, before viewing
an 18 km response. It must not be changed in response to favorable or
unfavorable electromagnetic results.

## Hardware Contract

The publication workflow uses the following fixed resource contract:

- Supported physical memory: 20 GB.
- Capacity reserved for Windows, WSL, filesystem cache, and monitoring: 6 GB.
- Maximum estimated memory for one formal solver run: 14 GB.
- FEniCSx and COMSOL formal solves must never overlap.
- Only one formal FEniCSx convergence run may execute at a time.
- empymod and lightweight report generation may run only when they do not
  overlap a solver initialization or materially reduce the 6 GB reserve.

The study runner must expose the total-memory and reserve values explicitly,
derive the 14 GB solver budget from them, pass that budget to the FEniCSx
pipeline, and record all three values in machine-readable preflight evidence.
The runner must reject nonfinite values, nonpositive total memory, negative
reserve, or a reserve greater than or equal to total memory.

## Revised Domain Axis

The domain axis is fixed as follows:

| Level | Horizontal extent | Earth depth | Air height | Source |
| --- | ---: | ---: | ---: | --- |
| small | 6000 m | 6000 m | 600 m | reuse completed 6 km, 0.5 percent run |
| standard | 12000 m | 12000 m | 1200 m | shared completed stage-two baseline |
| large | 18000 m | 18000 m | 1800 m | new run after memory preflight |

All domain levels retain the standard 8 m source-local target, 6 m
receiver-local target, 40 m and 30 m refinement radii, 750 m far-field target,
and 0.5 percent relative internal time-step limit through 1 ms. Source,
receiver, waveform, material, formulation, boundary condition, observation
times, KSP type, and KSP tolerances remain unchanged.

The coordinate contract remains physical `z=0` at ground, underground
positive, and air negative. The existing DOLFINx command conversion to its
earth-negative internal coordinates remains unchanged.

## Preflight And Data Flow

The formal workflow is fail-closed:

1. Resolve the paper-baseline levels and the 20/6/14 GB resource contract.
2. Generate the 18 km source-only mesh without starting time stepping.
3. Audit source coverage, receiver location, charge-conserving source
   divergence, mesh identity, nodes, retained cell blocks, tetrahedra, and
   estimated memory.
4. Write `preflight.json` with total memory, reserve, solver budget, estimate,
   mesh SHA256, and the existing source/receiver evidence.
5. Permit the forward command only if every scientific preflight passes and
   estimated memory is at most 14 GB.
6. Check live host memory before launch. Available physical memory must be at
   least the run estimate, and no COMSOL process may be running. The 6 GB
   operating-system reserve is already enforced by the independent 14 GB
   estimate ceiling and must not be added to the live-memory check a second
   time.
7. Run to 25 observation rows with a checkpoint every 10 internal steps.
8. Postprocess, evaluate all three convergence axes, and run the independent
   artifact audit.

An interrupted solve resumes only from a matching case manifest and checkpoint.
Changing the memory contract does not alter the physical case manifest, but the
exact resumed command and resource evidence must be rewritten for the current
contract.

## Failure Handling

If the 18 km estimate exceeds 14 GB, the runner must stop before forward
modeling and preserve the mesh and diagnostics. It must not automatically choose
a smaller domain, coarsen the far field, reduce local resolution, shorten the
time window, or loosen a numerical or publication threshold. Any further
resource redesign requires a new approved amendment based only on pre-solve
evidence.

If COMSOL starts during a FEniCSx run, interrupt FEniCSx cleanly, verify the
checkpoint and partial arrays, wait for COMSOL and its parent batch to exit,
verify free memory twice, and resume. External-process contention is an
environmental interruption, not a failed numerical result.

If the completed 18 km response fails a scientific gate, preserve and report
the failure. The 20 GB hardware limit must not be used to weaken the accepted
thresholds or omit late-time samples.

## Acceptance Gates

Time and local-mesh gates are unchanged:

- Standard-to-fine median relative difference at most 1 percent.
- Standard-to-fine RMS relative difference at most 2 percent.
- Standard-to-fine maximum relative difference at most 5 percent.
- Coarse-to-standard RMS at least standard-to-fine RMS minus a 0.1
  percentage-point tolerance.

Domain gates retain the approved logic with the revised physical labels:

- The 12-to-18 km RMS response change must be no greater than the 6-to-12 km
  RMS response change plus 0.1 percentage point.
- The 18 km run must pass the empymod gate over all 25 registered samples:
  median and RMS at most 5 percent and maximum at most 10 percent.

The candidate 12 km, 0.5 percent result is accepted for paper figures only when
the time, local-mesh, and revised domain axes all pass.

## Implementation Boundaries

This amendment changes only the FEniCSx layered convergence runner, level
definition, preflight metadata, tests, and associated reports. It does not
change the general FEniCSx solver, estimator coefficients, electromagnetic
equations, mesh generator, COMSOL model, or empymod reference.

COMSOL compliance with the same 20 GB total-memory requirement is a separate
subproject. It must define and verify a low-memory serial solve pathway before
the entire project is described as 20 GB compatible. Passing this amendment
alone establishes only that the layered FEniCSx convergence workflow meets the
contract.

## Tests And Verification

Before generating the 18 km mesh, automated tests must prove:

- The large domain is exactly 18/18/1.8 km and all non-domain parameters match
  the 12 km standard level.
- The default resource contract resolves to 20 GB total, 6 GB reserve, and
  14 GB solver budget.
- Custom valid contracts propagate consistently to pipeline arguments and
  preflight evidence.
- Invalid contracts fail before any subprocess is launched.
- An estimate above the derived budget is rejected, while an estimate equal to
  the budget is accepted.
- Existing checkpoint and remaining-output behavior is unchanged.

Real-run verification requires:

1. An 18 km `preflight.json` with `passed=true` and an estimate at most 14 GB.
2. A live launch audit showing no COMSOL process and available physical memory
   at least equal to the estimate.
3. Exactly 25 finite outputs, positive KSP convergence reasons, finite residuals,
   and a matching mesh hash.
4. The unchanged empymod and convergence gates evaluated from stored arrays.
5. Independent recomputation by `audit_layered_convergence.py`.
6. Focused and full related test suites passing after implementation.

## Publication Reporting

The manuscript and generated report must state that domain sensitivity was
tested at 6, 12, and 18 km under a 20 GB workstation contract. They must report
the exact mesh counts, estimator value, solver budget, response-change metrics,
and external-reference metrics. They must not imply that a 24 km response was
computed or that FEniCSx/COMSOL concurrency is supported.
