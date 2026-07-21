#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_ROOT [MESH_SEED_DIRECTORY]" >&2
  exit 2
fi

OUTPUT_ROOT=$(realpath -m "$1")
MESH_SEED_DIRECTORY=${2:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CONDA=${CONDA:-/home/paidaxin/miniconda3/bin/conda}
PYTHON=${PYTHON:-/home/paidaxin/miniconda3/envs/fenicsx/bin/python}
OUTPUT_INTERVAL_SUBSTEPS=${OUTPUT_INTERVAL_SUBSTEPS:-16}
MIN_STEPS_BEFORE_FIRST_OBSERVATION=${MIN_STEPS_BEFORE_FIRST_OBSERVATION:-1}
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OBSERVATION_TIMES=$(
  "$PYTHON" -c 'print(",".join(f"{10 ** (-5 + i / 10):.17g}" for i in range(51)))'
)

COMMON_ARGS=(
  --no-install
  --checkpoint-forward
  --memory-limit-gb 32
  --source-mode manual_line
  --source-projection-mode charge_conserving
  --source-rhs-sign -1
  --source-term-mode impressed_current
  --formulation e
  --initial-dc-mode fem
  --magnetic-receiver-mode faraday_integrated
  --magnetic-dbdt-mode curl
  --outer-boundary-mode pec
  --receiver-type point
  --receiver-average-radius 2
  --receiver-evaluation-mode median
  --rho-air 1e6
  --rho-earth 100
  --source-start-x -500
  --source-start-y 0
  --source-start-z -0.1
  --source-end-x 500
  --source-end-y 0
  --source-end-z -0.1
  --source-current 10
  --ramp-off-time 0
  --time-origin after_ramp
  --observation-times "$OBSERVATION_TIMES"
  --receiver-x 0
  --receiver-y -500
  --receiver-z -0.1
  --source-mesh-size 40
  --source-refinement-radius 100
  --receiver-mesh-size 20
  --receiver-refinement-radius 60
  --diffusion-refinement-mesh-size 80
  --nedelec-order 2
  --cole-n-terms 16
  --cole-f-min 1e-3
  --cole-f-max 1e4
  --cole-n-freq 81
  --time-method theta
  --time-theta 1
  --t-min 1e-6
  --t-max 1
  --min-steps-before-first-observation "$MIN_STEPS_BEFORE_FIRST_OBSERVATION"
  --output-interval-substeps "$OUTPUT_INTERVAL_SUBSTEPS"
  --x-extent 25000
  --y-extent 25000
  --air-height 25000
  --earth-depth 25000
  --layer-depths 300
  --layer-resistivities 100,100
)

seed_mesh() {
  local workdir=$1
  if [[ -z "$MESH_SEED_DIRECTORY" ]]; then
    return
  fi
  for name in \
    verification_mesh.msh \
    verification_mesh.dolfinx.msh \
    verification_mesh.msh.contract.json; do
    if [[ ! -e "$workdir/$name" ]]; then
      cp "$MESH_SEED_DIRECTORY/$name" "$workdir/$name"
    fi
  done
}

run_case() {
  local case_name=$1
  shift
  local workdir="$OUTPUT_ROOT/$case_name"
  mkdir -p "$workdir"
  seed_mesh "$workdir"

  local resume_args=()
  if [[ -e "$workdir/forward_checkpoint.npz" ]]; then
    resume_args=(--resume-forward)
  fi

  echo "[$(date --iso-8601=seconds)] starting $case_name" >> "$OUTPUT_ROOT/runner.log"
  /usr/bin/time -v \
    "$CONDA" run -n fenicsx --no-capture-output \
    python "$REPO_ROOT/dolfinx/sotem_pipeline.py" \
    --workdir "$workdir" \
    "${COMMON_ARGS[@]}" \
    "${resume_args[@]}" \
    "$@" \
    > "$workdir/run.log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed $case_name" >> "$OUTPUT_ROOT/runner.log"
}

mkdir -p "$OUTPUT_ROOT"
case "${RUN_CASES:-all}" in
  all)
    RUN_NOIP=true
    RUN_IP=true
    ;;
  noip)
    RUN_NOIP=true
    RUN_IP=false
    ;;
  ip)
    RUN_NOIP=false
    RUN_IP=true
    ;;
  *)
    echo "RUN_CASES must be one of: all, noip, ip" >&2
    exit 2
    ;;
esac

if $RUN_NOIP; then
  run_case song-noip-full --polarization none
fi
if $RUN_IP; then
  run_case song-ip-full \
    --polarization cole-cole \
    --cole-layer-top 0 \
    --cole-layer-bottom 300 \
    --cole-rho0 100 \
    --cole-m 0.3 \
    --cole-tau 1 \
    --cole-c 0.3
fi
