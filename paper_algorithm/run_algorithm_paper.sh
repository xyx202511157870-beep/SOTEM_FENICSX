#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-help}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/generated/paper_algorithm"}
CONDA=${CONDA:-conda}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-fenicsx}
MEMORY_LIMIT_GB=${MEMORY_LIMIT_GB:-32}
FORCE=${FORCE:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONDA_RUN=("$CONDA" run -n "$CONDA_ENV_NAME" --no-capture-output)
PYTHON_RUN=("${CONDA_RUN[@]}" "$PYTHON_BIN")

mkdir -p "$OUTPUT_ROOT"

obs_logspace() {
  local start_exp=$1
  local stop_exp=$2
  local count=$3
  "${PYTHON_RUN[@]}" -c \
    "import numpy as np; print(','.join(f'{v:.17g}' for v in np.logspace($start_exp,$stop_exp,$count)))"
}

run_logged() {
  local name=$1
  shift
  local workdir="$OUTPUT_ROOT/$name"
  mkdir -p "$workdir"
  if [[ "$FORCE" != "1" ]] && [[ -f "$workdir/verification_report.txt" || -f "$workdir/validation_summary.json" ]]; then
    echo "[skip] $name already has a completed artifact. Set FORCE=1 to rerun."
    return
  fi
  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$OUTPUT_ROOT/runner.log"
  {
    /usr/bin/time -v "$@"
  } >"$workdir/run.log" 2>&1
  echo "[$(date --iso-8601=seconds)] DONE  $name" | tee -a "$OUTPUT_ROOT/runner.log"
}

run_benchmark() {
  local label=$1
  local case_file=$2
  local variant=$3
  local level=$4
  local workdir="$OUTPUT_ROOT/$label"
  run_logged "$label" \
    "${PYTHON_RUN[@]}" "$REPO_ROOT/dolfinx/run_sotem_benchmark.py" \
    --case "$REPO_ROOT/$case_file" \
    --variant "$variant" \
    --level "$level" \
    --workdir "$workdir" \
    --no-install
}

run_direct() {
  local label=$1
  shift
  local workdir="$OUTPUT_ROOT/$label"
  local resume_args=()
  if [[ -f "$workdir/forward_checkpoint.npz" ]]; then
    resume_args=(--resume-forward)
  fi
  run_logged "$label" \
    "${PYTHON_RUN[@]}" "$REPO_ROOT/dolfinx/sotem_pipeline.py" \
    --workdir "$workdir" \
    --no-install \
    --checkpoint-forward \
    --memory-limit-gb "$MEMORY_LIMIT_GB" \
    "${resume_args[@]}" \
    "$@"
}

run_env() {
  "${PYTHON_RUN[@]}" "$REPO_ROOT/dolfinx/run_sotem_benchmark.py" \
    --case "$REPO_ROOT/benchmarks/sotem/lei2023_noip.yaml" \
    --variant noip \
    --level S0T0B0 \
    --workdir "$OUTPUT_ROOT/env_check" \
    --check-env-only \
    --no-install
  "${PYTHON_RUN[@]}" -m pytest -q \
    "$REPO_ROOT/tests/test_sources.py" \
    "$REPO_ROOT/tests/test_receivers.py" \
    "$REPO_ROOT/tests/test_empymod_magnetic6.py" \
    "$REPO_ROOT/tests/test_empymod_waveform.py" \
    "$REPO_ROOT/tests/test_materials_cole_cole.py"
}

run_preflight() {
  local workdir="$OUTPUT_ROOT/preflight_lei"
  run_logged "preflight_lei" \
    "${PYTHON_RUN[@]}" "$REPO_ROOT/dolfinx/run_sotem_benchmark.py" \
    --case "$REPO_ROOT/benchmarks/sotem/lei2023_noip.yaml" \
    --variant noip \
    --level S0T0B0 \
    --workdir "$workdir" \
    --source-only \
    --no-install
}

run_benchmark_pilot() {
  run_benchmark "v1_lei_noip_S0T0B0" "benchmarks/sotem/lei2023_noip.yaml" noip S0T0B0
  run_benchmark "v2_zhou_noip_S0T0B0" "benchmarks/sotem/zhou2020_grounded_wire.yaml" noip S0T0B0
  run_benchmark "v5_zhou_ip_S0T0B0" "benchmarks/sotem/zhou2020_grounded_wire.yaml" ip S0T0B0
}

run_convergence() {
  local levels=(S0T0B0 S1T0B0 S2T0B0 S2T1B0 S2T2B0 S2T2B1 S2T2B2)
  for level in "${levels[@]}"; do
    run_benchmark "v1_lei_noip_${level}" "benchmarks/sotem/lei2023_noip.yaml" noip "$level"
  done
  run_benchmark "v2_zhou_noip_S2T2B2" "benchmarks/sotem/zhou2020_grounded_wire.yaml" noip S2T2B2
}

magnetic6_common_args() {
  local ramp=$1
  local obs=$2
  cat <<EOF
--source-mode
manual_line
--source-projection-mode
charge_conserving
--source-rhs-sign
-1
--source-term-mode
impressed_current
--formulation
e
--initial-dc-mode
fem
--magnetic-receiver-mode
biot_current
--magnetic-dbdt-mode
curl
--magnetic-output-contract
magnetic6
--magnetic-recovery-quadrature-degree
8
--magnetic-recovery-quadrature-audit-degrees
4,6,8,10
--outer-boundary-mode
pec
--receiver-type
point
--receiver-evaluation-mode
median
--rho-air
1e8
--rho-earth
100
--source-start-x
-20
--source-start-y
-7
--source-start-z
-0.1
--source-end-x
20
--source-end-y
7
--source-end-z
-0.1
--source-current
1
--ramp-off-time
$ramp
--time-origin
after_ramp
--observation-times
$obs
--receiver-x
13
--receiver-y
31
--receiver-z
-0.2
--source-mesh-size
1
--source-refinement-radius
20
--receiver-mesh-size
0.5
--receiver-refinement-radius
15
--diffusion-refinement-mesh-size
20
--nedelec-order
2
--expected-source-length
42.37924020083418
--expected-parallel-offset
24.96505352588116
--wire-radius
0.5
--time-method
theta
--time-theta
1
--t-min
1e-6
--t-max
1e-2
--ramp-solver-t-min
5e-7
--min-steps-during-turnoff
10
--min-steps-before-first-observation
4
--output-interval-substeps
4
--x-extent
3000
--y-extent
3000
--air-height
1500
--earth-depth
3000
--empymod-srcpts
9
--reference-audit-srcpts
17
--rtol
1e-8
--atol
1e-12
--max-it
1000
EOF
}

run_magnetic6_case() {
  local label=$1
  local ramp=$2
  local yaml=$3
  local obs
  obs=$(obs_logspace -5 -2 31)
  mapfile -t args < <(magnetic6_common_args "$ramp" "$obs")
  run_direct "$label" "${args[@]}"

  local workdir="$OUTPUT_ROOT/$label"
  local validation_dir="$workdir/empymod_magnetic6"
  local validator_args=(
    "$REPO_ROOT/$yaml"
    --numerical "$workdir/magnetic6_numerical.npz"
    --depths 0
    --resistivities 1e8,100
    --srcpts 9
    --srcpts-audit 17
    --audit-tolerance 0.01
    --comparison-tolerance 0.05
    --output-dir "$validation_dir"
  )
  if [[ "$ramp" != "0" ]]; then
    validator_args+=(--waveform-quadrature-order 8)
  else
    validator_args+=(--ideal-step-off)
  fi
  run_logged "${label}_validator" \
    "${CONDA_RUN[@]}" atem3d-validate-empymod-magnetic6 "${validator_args[@]}"
}

run_magnetic6() {
  run_magnetic6_case \
    "v3_magnetic6_step" \
    0 \
    "examples/empymod_validation_magnetic6.yaml"
  run_magnetic6_case \
    "v3_magnetic6_ramp5us" \
    5e-6 \
    "examples/empymod_validation_magnetic6_5us.yaml"
}

run_waveform() {
  run_magnetic6
  local obs
  obs=$(obs_logspace -5 -2 31)
  mapfile -t args20 < <(magnetic6_common_args 2e-5 "$obs")
  run_direct "v3_magnetic6_ramp20us" "${args20[@]}"
}

run_receiver() {
  local obs
  obs=$(obs_logspace -5 -2 31)
  for radius in 0.5 1 2 4; do
    mapfile -t args < <(magnetic6_common_args 5e-6 "$obs")
    args+=(
      --receiver-diagnostic-types disk_average
      --receiver-average-radius "$radius"
    )
    run_direct "v4_receiver_disk_r${radius}m" "${args[@]}"
  done
}

run_ip() {
  run_logged "v5_ip_debye_sweep" \
    "${PYTHON_RUN[@]}" "$REPO_ROOT/paper_algorithm/run_ip_debye_sweep.py" \
    --output-root "$OUTPUT_ROOT/v5_ip_debye_sweep" \
    --level "${IP_LEVEL:-S1T1B1}" \
    --terms "${IP_TERMS:-8,12,16,20}" \
    --memory-limit-gb "$MEMORY_LIMIT_GB"
}

run_demo3d() {
  run_logged "v6_3d_channel_demo" \
    "${PYTHON_RUN[@]}" "$REPO_ROOT/paper_algorithm/run_3d_channel_demo.py" \
    --output-root "$OUTPUT_ROOT/v6_3d_channel_demo" \
    --profile "${DEMO3D_PROFILE:-pilot}"
}

case "$MODE" in
  env) run_env ;;
  preflight) run_preflight ;;
  benchmark-pilot) run_benchmark_pilot ;;
  convergence) run_convergence ;;
  magnetic6) run_magnetic6 ;;
  waveform) run_waveform ;;
  receiver) run_receiver ;;
  ip) run_ip ;;
  demo3d) run_demo3d ;;
  all)
    run_env
    run_preflight
    run_benchmark_pilot
    run_magnetic6
    run_convergence
    run_waveform
    run_receiver
    run_ip
    run_demo3d
    ;;
  help|-h|--help)
    cat <<EOF
Usage:
  bash paper_algorithm/run_algorithm_paper.sh MODE

Modes:
  env               environment and unit checks
  preflight         mesh/source/memory preflight only
  benchmark-pilot   coarse no-IP/layered/IP checks
  magnetic6         ideal and 5 us six-component validation
  convergence       seven-stage no-IP convergence + layered model
  waveform          ideal, 5 us and 20 us waveform cases
  receiver          point versus finite-disk receiver diagnostics
  ip                Zhou-2020 Debye-term sweep
  demo3d            3-D tortuous conductive-path self-convergence
  all               run every stage

Environment variables:
  OUTPUT_ROOT, CONDA_ENV_NAME, MEMORY_LIMIT_GB, FORCE
  IP_LEVEL, IP_TERMS, DEMO3D_PROFILE
EOF
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
