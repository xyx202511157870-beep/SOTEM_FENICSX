# Canonical SimPEG PETSc Initialization Design

## Goal

Remove the canonical S0 sparse-direct initialization bottleneck without changing
the physical equations, gauge, source, time grid, signs, amplitudes, or acceptance
tolerances. Keep the existing direct initialization as the default compatibility
path for noncanonical and small configurations.

## Confirmed root cause

The failed Lei no-IP S0 run reached the 20 GiB cgroup limit while the run manifest
was still `prepared` and before any transient diagnostics were published. The
initial DC electric solve factors the gauge-reduced nodal matrix
`G.T M_sigma0 G`; the Ampere-consistent magnetic initialization then factors the
larger edge matrix `C.T M_mu^-1 C + G G.T`. The latter fill-in is the immediate
large-memory failure surface. The transient PETSc/HYPRE AMS solver had not yet
been reached.

## Linear systems and constraints

The DC system retains the existing one-node scalar-potential gauge. With positive
conductivity and a connected tensor mesh, the unreduced graph Laplacian is
positive semidefinite with the constant nullspace and the reduced system is SPD.
It will use PETSc KSP with HYPRE BoomerAMG. Acceptance requires a positive PETSc
convergence reason, finite solution, and SciPy-evaluated
`||rhs - A*x||/||rhs|| <= 1e-8`. The reconstructed edge current must also satisfy
the discrete divergence balance.

The magnetic system retains the existing Coulomb-style `G G.T` gauge term. Its
curl-curl and gauge terms are complementary positive-semidefinite operators on a
simply connected tensor mesh, yielding an SPD edge H(curl) system. It will use
PETSc KSP with HYPRE AMS, the full tensor-mesh discrete gradient, and discrete
constant edge vectors. Acceptance requires the same independent residual gate
and the reconstructed magnetic flux must satisfy the Ampere balance.

## Configuration and provenance

`TDEMIPSimulation` gains an explicit initialization solver selector. Its default
remains `direct`. The canonical SOTEM adapter explicitly selects the PETSc/HYPRE
initialization path and truthfully records the scalar BoomerAMG and edge AMS
backends. Initialization diagnostics are returned separately from transient
time-step diagnostics and include phase, backend reason, iterations, external
true residual, tolerance, and balance residual.

## Lifetime and failure behavior

Each initialization solve creates its native PETSc solver locally and destroys it
in `finally`, including convergence-gate and construction failures. A nonfinite
solution, nonpositive backend reason, excessive external residual, or excessive
physical balance residual fails closed. The adapter does not publish a response
after any initialization failure.

## Validation

Tests first cover canonical selection and provenance, direct compatibility,
external residual and balance gates, and cleanup. Real WSL tests compare the new
path to direct on a small tensor mesh, then exercise a medium tensor mesh while
recording true residuals and peak memory. The 99,840-cell S0 case is explicitly
excluded from this change's verification run.
