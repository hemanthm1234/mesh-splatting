#!/bin/bash
set -e

SCENE=${1:-"Truck"}
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
DATASET_DIR="/data1/hemanth/datasets/tandt/$SCENE"
OUTPUT_DIR="$EXP_DIR/output/${SCENE}_baseline"

# Toggle this for test vs full run
NUM_ITERATIONS=30000

# Safety check to prevent overwriting an existing training run!
if [ -d "$OUTPUT_DIR" ]; then
    echo "[!] Error: $OUTPUT_DIR already exists."
    echo "    To run a new experiment for $SCENE, remove the directory or rename it."
    exit 1
fi

echo "=== Baseline MeshSplatting Metrics: T&T $SCENE ==="

# 0. WandB Setup
if [ -z "$WANDB_API_KEY" ]; then
    echo "[!] WANDB_API_KEY is not set. WandB will run in offline/anonymous mode."
else
    echo "[0/6] WANDB_API_KEY detected! Authenticating with Weights & Biases..."
    wandb login "$WANDB_API_KEY"
fi

# 1. Download Dataset (Phase 0) - Skipped since dataset is manually processed
echo "[1/6] Dataset is manually prepared for $SCENE."

# 1. Generate Depth Maps (Phase 1) - Skipped since depth maps are pre-computed offline
echo "[2/6] Depth maps are pre-computed offline."
# bash "$EXP_DIR/generate_depth.sh" "$SCENE"

# 3. Training (Phase 3)
bash "$EXP_DIR/03_train.sh" "$SCENE"

# 4. Rendering & Metrics (Phase 4)
bash "$EXP_DIR/04_render_metrics.sh" "$SCENE"

# 5. Extract Native Mesh (Phase 5)
bash "$EXP_DIR/06_extract_native.sh" "$SCENE" $NUM_ITERATIONS

# 5.5. Evaluate T&T Baseline metrics
# Note: The mapping reference file warning has been removed as it's not needed for static scenes
bash "$EXP_DIR/06_b_eval_tandt.sh" "$SCENE" $NUM_ITERATIONS || echo "[!] Evaluation failed. Continuing pipeline..."

# 6. Create Video (Phase 6)
bash "$EXP_DIR/07_create_video.sh" "$SCENE" $NUM_ITERATIONS

# 7. Store & Log Results (aggregates all metrics → central JSON + WandB summary)
echo "=== [Step 8] Storing & Logging Results for $SCENE ==="
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
python "$EXP_DIR/store_results.py"
echo "=== Results Stored! ==="

echo "=== Experiment Complete for $SCENE! ==="
