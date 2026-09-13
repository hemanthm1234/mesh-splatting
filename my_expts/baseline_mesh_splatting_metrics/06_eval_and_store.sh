#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"Truck"}
NUM_ITERATIONS=${2:-30000}

echo "=== [Step 6] Running Surface Eval and Storing Results ==="
cd "$EXP_DIR"
# bash 06_b_eval_tandt.sh "$SCENE" $NUM_ITERATIONS # Skipped: Chamfer Distance is incompatible with T&T native mesh
python store_results.py
echo "=== Evaluation Complete! ==="
