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
equation and the receiver Faraday state.  Use backward Euler through the first
reported observation, then switch both states to BDF2 on the following
internal step.  Keep all backward-Euler and Cole--Cole behavior unchanged;
Cole--Cole remains restricted to backward Euler.

This longer startup replaces the original single-step startup after runtime
validation disproved that assumption.  On the unchanged Song no-IP P2/T4
case, backward Euler gave first-observation relative errors of 2.36% in `Ex`,
0.06% in `Hz`, and 1.49% in `dBzdt`; BDF2 after only one backward-Euler step
gave 78.72%, 0.61%, and 52.23%, respectively.  The magnetic initial field
remained accurate, while the electric and derivative responses overshot.
This identifies startup damping, not the Faraday recurrence or `H0`, as the
remaining first-observation defect.

Alternatives rejected for this validation cycle:

- Increasing backward Euler from T4 to T16 for the whole time window is
  scientifically valid but costs roughly four times as many solves and remains
  first order.  The selected warmup pays this damping cost only before the
  first reported observation.
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

The forward loop retains the previous two electric and receiver `Hz` states
whenever `time_method=bdf2`.  Let `first_output_step` be the first entry of
`schedule["output_step_indices"]`.  Steps satisfying
`step <= first_output_step` use backward Euler.  BDF2 begins only when
`step > first_output_step` and complete older-state/time-step history exists.
For the current schedule, steps 0--15 are backward Euler and step 16 is the
first BDF2 step.  BDF2 checkpoint resume remains prohibited by the existing
model-consistency gate, so no checkpoint schema change is required.

## Failure handling and provenance

- Reject non-finite/non-positive permeability and incomplete BDF2 state.
- Do not fall back silently from BDF2 to backward Euler after the scheduled
  warmup.
- Continue logging the actual per-step time method and coefficients through
  the existing solver metadata.
- Preserve the failed BE/T4 artifacts and do not alter error thresholds,
  observation times, component selection, or reference data.

## Verification

1. Add a unit test for the startup selector: BDF2 is disabled through the
   first output step, enabled on the following step, and never enabled for a
   theta run.
2. Verify the constant-step BDF2 recurrence against its closed-form value.
3. Verify invalid or incomplete BDF2 history is rejected.
4. Run the focused Faraday, model-consistency, partial-forward, and operator
   cache tests, followed by the full WSL test suite.
5. Run the unchanged Song no-IP P2 mesh with BDF2 and T4 output substeps.
   Formal acceptance continues to require `Ex`, `Hz`, and `dBzdt` over all 51
   times to satisfy the existing 5% gate; `Ey` remains diagnostic-only.
6. Only after this no-IP gate passes, run the current-version SimPEG no-IP
   case and then resume paired IP/no-IP validation.
