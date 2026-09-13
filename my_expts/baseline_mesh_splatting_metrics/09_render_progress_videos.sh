#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting

EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
OUTPUT_DIR="$EXP_DIR/output/truck_v3"
PROGRESS_DIR="$OUTPUT_DIR/progress"

echo "=== [Step 9] Rendering Progress Videos ==="
cd "$EXP_DIR" || exit 1

python render_progress_videos.py --progress_dir "$PROGRESS_DIR" --fps 5

echo "=== Progress Videos Complete! ==="
