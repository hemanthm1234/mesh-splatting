#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"truck"}
DATASET_DIR="/data1/hemanth/datasets/tandt/$SCENE"
OUTPUT_DIR="$EXP_DIR/output/${SCENE}_baseline"
NUM_ITERATIONS=30000

echo "=== [Step 4] Rendering & Metrics ==="
cd /data1/hemanth/mesh-splatting
python render.py \
    --iteration $NUM_ITERATIONS \
    -s "$DATASET_DIR" \
    -m "$OUTPUT_DIR" \
    --eval \
    --skip_train \
    --quiet

echo "Computing Metrics..."
python metrics.py -m "$OUTPUT_DIR"
echo "=== Rendering & Metrics Complete! ==="
