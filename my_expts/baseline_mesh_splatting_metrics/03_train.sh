#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"truck"}
DATASET_DIR="/data1/hemanth/datasets/tandt/$SCENE"
OUTPUT_DIR="$EXP_DIR/output/${SCENE}_baseline"
NUM_ITERATIONS=30000

echo "=== [Step 3] Training ==="
if [ -z "$WANDB_API_KEY" ]; then
    echo "[!] WANDB_API_KEY is not set. WandB will run in offline/anonymous mode."
else
    wandb login "$WANDB_API_KEY"
fi

cd /data1/hemanth/mesh-splatting
python train.py \
    -s "$DATASET_DIR" \
    -m "$OUTPUT_DIR" \
    --iterations $NUM_ITERATIONS \
    --quiet \
    --eval \
    --scene_name "$SCENE" \
    --wandb_name "${SCENE,,}_1"
echo "=== Training Complete! ==="
