# Faraday BDF2 consistency design

## Context

The corrected high-order Biot--Savart initial field removes the dominant
constant magnetic offset.  The no-IP P2/T4 backward-Euler run nevertheless
first exceeds the unchanged 5% physical gate in `Hz` at 3.162 ms while `Ex`
and `dBzdt` remain below the gate.  The existing solver also exposes a BDF2
electric-field method, but receiver `Hz` is always advanced with the
backward-Euler formula.  Selecting BDF2 therefore makes the electric and
magnetic state updates discretely inconsistent.

## Decision

Use the existing variable-step BDF2 coefficients for both the electric-field
equation and the receiver Faraday state.  Keep the first step as the existing
backward-Euler startup step.  Keep all backward-Euler and Cole--Cole behavior
unchanged; Cole--Cole remains restricted to backward Euler.

Alternatives rejected for this validation cycle:

- Increasing backward Euler from T4 to T16 is scientifically valid but costs
  roughly four times as many solves and remains first order.
- Crank--Nicolson would also require a matching trapezoidal Faraday update and
  is more exposed to step-off startup oscillations.  It is not the selected
  production validation path.

## State and update

For variable-step BDF2, reuse `_bdf2_step_coefficients(dt, previous_dt)`:

```text
alpha0 B_n - beta1 B_(n-1) - beta2 B_(n-2) = dBdt_n
```

where the helper returns `lhs=alpha0`, `old=beta1`, and `older=beta2`
(`older` is negative).  At the receiver,

```text
H_n = (dBdt_n / mu + old * H_(n-1) + older * H_(n-2)) / lhs
```

The forward loop retains the previous two receiver `Hz` states whenever
`time_method=bdf2`.  Step zero uses the existing backward-Euler update; after
that step the initial `H0` becomes the older state.  BDF2 checkpoint resume
remains prohibited by the existing model-consistency gate, so no checkpoint
schema change is required.

## Failure handling and provenance

- Reject non-finite/non-positive permeability and incomplete BDF2 state.
- Do not fall back silently from BDF2 to backward Euler after startup.
- Continue logging the actual per-step time method and coefficients through
  the existing solver metadata.
- Preserve the failed BE/T4 artifacts and do not alter error thresholds,
  observation times, component selection, or reference data.

## Verification

1. Add a unit test that requests a BDF2 Faraday update and currently fails
   because the helper/API does not exist.
2. Verify the constant-step BDF2 recurrence against its closed-form value.
3. Verify invalid or incomplete BDF2 history is rejected.
4. Run the focused Faraday, model-consistency, partial-forward, and operator
   cache tests, followed by the full WSL test suite.
5. Run the unchanged Song no-IP P2 mesh with BDF2 and T4 output substeps.
   Formal acceptance continues to require `Ex`, `Hz`, and `dBzdt` over all 51
   times to satisfy the existing 5% gate; `Ey` remains diagnostic-only.
6. Only after this no-IP gate passes, run the current-version SimPEG no-IP
   case and then resume paired IP/no-IP validation.
