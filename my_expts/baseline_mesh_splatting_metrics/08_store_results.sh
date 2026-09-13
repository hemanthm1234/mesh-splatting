#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"

echo "=== [Step 8] Storing Aggregated Results ==="
cd "$EXP_DIR"

python store_results.py

echo "=== Results Stored! ==="
