#!/bin/bash
set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SCENES=("Truck" "Meeting_room" "Caterpillar" "Ignatius" "Barn" "Church")
BASE_DATASET_DIR="/data1/hemanth/datasets/tandt"
OUTPUT_BASE_DIR="./my_expts/vggt_omega_initialised/output_full_5M"

mkdir -p "${OUTPUT_BASE_DIR}"

for SCENE in "${SCENES[@]}"; do
    echo "=================================================="
    echo "Starting VGGT Full Training for scene: ${SCENE}"
    echo "=================================================="
    if python train.py \
        --source_path "${BASE_DATASET_DIR}/${SCENE}" \
        --model_path "${OUTPUT_BASE_DIR}/${SCENE}" \
        --vggt_mode full_pipeline \
        --vggt_ply_name "vggt_omega_5M.ply" \
        --eval \
        --wandb_name "vggt_full_5M_${SCENE}"; then
        echo "✅ Finished VGGT Full Training for scene: ${SCENE}"
    else
        echo "❌ Error encountered during VGGT Full Training for scene: ${SCENE}. Continuing to next scene..."
    fi
done

echo "🎉 Finished executing VGGT full training script for all scenes!"

