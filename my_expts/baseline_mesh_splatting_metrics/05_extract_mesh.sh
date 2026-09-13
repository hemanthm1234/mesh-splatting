#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
DATASET_DIR="/data1/hemanth/datasets/tandt/truck"
OUTPUT_DIR="$EXP_DIR/output/truck"
NUM_ITERATIONS=30000

echo "=== [Step 5] Extracting TSDF Mesh ==="
cd /data1/hemanth/mesh-splatting
python mesh.py \
    -s "$DATASET_DIR" \
    -m "$OUTPUT_DIR" \
    --iteration $NUM_ITERATIONS \
    --unbounded
echo "=== Mesh Extraction Complete! ==="
