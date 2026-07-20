#!/usr/bin/env bash
set -euo pipefail

source /home/paidaxin/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx

cd "$(dirname "$0")/.."
python dolfinx/sotem_pipeline.py \
  --workdir dolfinx/analyticdc_small_smoke \
  --force-mesh \
  --initial-dc-mode analytic_halfspace \
  --rho-earth 200 \
  --t-min 1e-5 \
  --t-max 1.1e-5 \
  --time-growth 1.1 \
  --x-extent 800 \
  --y-extent 300 \
  --air-height 300 \
  --earth-depth 500 \
  --source-mesh-size 25 \
  --source-refinement-radius 60 \
  --receiver-mesh-size 25 \
  --receiver-refinement-radius 60 \
  --memory-limit-gb 32 \
  --memory-safety-fraction 0.95 \
  --magnetic-receiver-mode faraday_integrated \
  --empymod-srcpts 17 \
  --reference-audit-srcpts 33 \
  --max-it 300 \
  --rtol 1e-7
