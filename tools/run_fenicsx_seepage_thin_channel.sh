#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec /home/paidaxin/miniconda3/envs/fenicsx/bin/python -u dolfinx/sotem_pipeline.py \
  --workdir output/seepage_channel_100m_5rx_60x1x1/fenicsx_channel \
  --no-install \
  --x-extent 6000 --y-extent 6000 --air-height 600 --earth-depth 6000 \
  --source-start-x -50 --source-start-y 0 --source-start-z -0.1 \
  --source-end-x 50 --source-end-y 0 --source-end-z -0.1 \
  --source-current 1 --expected-source-length 100 --expected-parallel-offset 0 \
  --receiver-x 0 --receiver-y 0 --receiver-z 0.1 \
  --receiver-location 0,-20,0.1 \
  --receiver-location 0,-10,0.1 \
  --receiver-location 0,0,0.1 \
  --receiver-location 0,10,0.1 \
  --receiver-location 0,20,0.1 \
  --source-mesh-size 5 --source-refinement-radius 40 \
  --receiver-mesh-size 2.5 --receiver-refinement-radius 20 \
  --far-field-mesh-size 750 \
  --conductivity-box-name seepage_channel \
  "--conductivity-box-bounds=-30,30;-0.5,0.5;-20.5,-19.5" \
  --conductivity-box-sigma 1.0 \
  --conductivity-box-mesh-size 0.25 \
  --rho-air 1e8 --rho-earth 100 \
  --ramp-off-time 1e-8 --time-origin after_ramp \
  --t-min 1e-5 --t-max 1e-2 --time-growth 1.2589254117941673 \
  --max-internal-dt 1e-4 --max-internal-dt-fraction 0.05 \
  --max-internal-dt-fraction-until 1e-3 \
  --min-steps-during-turnoff 10 --min-steps-before-first-observation 20 \
  --time-method theta --time-theta 1 --outer-boundary-mode natural \
  --formulation e --initial-dc-mode fem \
  --magnetic-receiver-mode biot_current --magnetic-dbdt-mode curl \
  --receiver-evaluation-mode median \
  --ksp-type cg --rtol 1e-7 --atol 1e-12 --max-it 4000 \
  --empymod-srcpts 257 --reference-audit-srcpts 513 \
  --memory-limit-gb 30 --memory-safety-fraction 0.95 \
  --checkpoint-forward \
  "$@"
