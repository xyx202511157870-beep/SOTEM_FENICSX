# Remove COMSOL From the Seepage Report Design

## Scope

The formal seepage-channel result set contains SimPEG, FEniCSx, and empymod only. COMSOL runner code and Git history remain available for possible future work, but no COMSOL calculation, file, gate, curve, legend, caption, conclusion, or evidence path is part of this report release.

## Scientific roles

- SimPEG solves the 3D background, finite 60 x 1 x 1 m conductivity channel, parameter sweeps, and convergence cases.
- FEniCSx independently solves the same full 3D domain. It must not solve one side and mirror the result.
- empymod is a layered uniform-background reference only. It is not presented as a solver for the finite 3D channel.
- Formal receivers remain Rx1, Rx2, Rx4, and Rx5. Rx3 remains excluded from formal curves and cross-solver metrics.

## Verification contract

The final summary is the already-defined open-3D summary. Every gate in `OPEN3D_REQUIRED_GATES` must be available and pass before plots or Word can be generated. Missing or failed gates remain fail-closed. The summary records the canonical model fingerprint and must not read `comsol_3d`.

## Figures and report

The formal figure set contains z-down model geometry, conventional absolute-value decay curves, signed and relative channel anomalies, parameter sweeps, convergence, parity, and a two-solver 3D anomaly comparison for SimPEG and FEniCSx. The obsolete three-solver anomaly figure is removed.

The Word cover, solver table, scientific conclusions, limitations, evidence table, and figure captions name only the three retained algorithms. The report explicitly states that empymod validates the uniform background and that the finite 3D anomaly comparison is SimPEG versus FEniCSx. It also states that COMSOL is outside this release scope.

## Cleanup and compatibility

All COMSOL output and scratch data for this task are deleted. The standalone COMSOL implementation and tests remain in the repository, but the formal aggregation, plotting, and Word-building entry points no longer depend on them. Existing non-COMSOL numerical outputs are preserved.

## Acceptance evidence

1. Unit tests prove the final summary passes without a `comsol_3d` directory and has no COMSOL gates.
2. Plot tests prove the formal figure manifest and generated legends contain no COMSOL or obsolete three-solver artifact.
3. Word tests prove report text contains no COMSOL solver result claims and includes the explicit scope statement.
4. The real final aggregation passes with `--require-pass`.
5. All formal figures regenerate from passing data.
6. The Word file regenerates only after the gates pass and is rendered page-by-page for visual inspection.

