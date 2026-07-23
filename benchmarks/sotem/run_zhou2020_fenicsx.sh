#!/usr/bin/env bash
set -euo pipefail

variant="${1:-noip}"
level="${2:-S0T0B0}"
workdir="${3:-generated/validation/zhou2020_grounded_wire/${variant}-${level}}"

python dolfinx/run_sotem_benchmark.py \
  --case benchmarks/sotem/zhou2020_grounded_wire.yaml \
  --variant "${variant}" \
  --level "${level}" \
  --workdir "${workdir}" \
  --no-install
