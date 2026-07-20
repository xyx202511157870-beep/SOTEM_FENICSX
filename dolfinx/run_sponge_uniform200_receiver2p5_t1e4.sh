#!/usr/bin/env bash
set -euo pipefail

source /home/paidaxin/miniconda3/etc/profile.d/conda.sh
conda activate fenicsx

cd /mnt/d/Doctor/codex_app/simpeg自编时域电性源瞬变电磁法求解

python dolfinx/sotem_pipeline.py \
  --workdir dolfinx/sponge_uniform200_receiver2p5_t1e3_meshonly \
  --t-min 1e-5 \
  --t-max 1e-4 \
  --time-growth 1.1 \
  --x-extent 8000 \
  --y-extent 8000 \
  --air-height 5000 \
  --earth-depth 8000 \
  --layer-depths 2000,2200 \
  --layer-resistivities 200,200,200 \
  --source-mesh-size 5 \
  --source-refinement-radius 100 \
  --receiver-mesh-size 2.5 \
  --receiver-refinement-radius 80 \
  --diffusion-refinement-factor 2.0 \
  --diffusion-refinement-mesh-size 80 \
  --sponge-strength 0.01 \
  --sponge-thickness 1500 \
  --sponge-sides x_min,x_max,y_min,y_max,z_min,z_max \
  --magnetic-receiver-mode faraday_integrated \
  --empymod-srcpts 65 \
  --reference-audit-srcpts 129 \
  --max-it 1000 \
  --rtol 1e-8
