#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"truck"}
DATASET_DIR="/data1/hemanth/datasets/tandt/$SCENE"
OUTPUT_DIR="$EXP_DIR/output/${SCENE}_baseline"

NUM_ITERATIONS=${2:-30000}

echo "=== [Step 7] Creating Video ==="
cd /data1/hemanth/mesh-splatting

if [ ! -d "$OUTPUT_DIR/point_cloud/iteration_${NUM_ITERATIONS}" ]; then
    echo "Error: Checkpoint iteration_${NUM_ITERATIONS} not found. Cannot create video."
    exit 1
fi

python create_video.py \
    -s "$DATASET_DIR" \
    -m "$OUTPUT_DIR" \
    --iteration $NUM_ITERATIONS \
    --save_as "rendered_video_${NUM_ITERATIONS}"

echo "=== Video Creation Complete! ==="
echo "Video saved as $OUTPUT_DIR/rendered_video_${NUM_ITERATIONS}.mp4"
