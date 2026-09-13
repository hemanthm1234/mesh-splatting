#!/bin/bash
set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SCENES=("Truck")
BASE_DATASET_DIR="/data1/hemanth/datasets/tandt"
OUTPUT_BASE_DIR="./my_expts/vggt_omega_initialised/output_full_5M_noDP"

mkdir -p "${OUTPUT_BASE_DIR}"

for SCENE in "${SCENES[@]}"; do
    echo "=================================================="
    echo "Starting VGGT Training (NO DENSIFICATION OR PRUNING) for scene: ${SCENE}"
    echo "=================================================="
    if python train.py \
        --source_path "${BASE_DATASET_DIR}/${SCENE}" \
        --model_path "${OUTPUT_BASE_DIR}/${SCENE}" \
        --vggt_mode full_pipeline \
        --vggt_ply_name "vggt_omega_5M.ply" \
        --eval \
        --start_pruning 30000 \
        --densify_from_iter 30000 \
        --wandb_name "vggt_5M_noDP_${SCENE}"; then
        echo "✅ Finished VGGT Training for scene: ${SCENE}"
    else
        echo "❌ Error encountered during VGGT Training for scene: ${SCENE}."
    fi
done

echo "🎉 Finished executing script!"
