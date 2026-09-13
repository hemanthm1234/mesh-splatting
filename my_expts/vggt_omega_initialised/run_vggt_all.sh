#!/bin/bash
set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SCENES=("Truck" "Ignatius" "Meeting_room" "Caterpillar" "Barn")
BASE_DATASET_DIR="/data1/hemanth/datasets/tandt"

DIR_BASE="./my_expts/vggt_omega_initialised/output_colmap_base"
DIR_NODP="./my_expts/vggt_omega_initialised/output_colmap_noDP"
DIR_NOPRUNE="./my_expts/vggt_omega_initialised/output_colmap_noPrune"
DIR_NODENSIFY="./my_expts/vggt_omega_initialised/output_colmap_noDensify"

mkdir -p "${DIR_BASE}" "${DIR_NODP}" "${DIR_NOPRUNE}" "${DIR_NODENSIFY}"

for SCENE in "${SCENES[@]}"; do
    echo "=================================================="
    echo "Processing Scene: ${SCENE}"
    echo "=================================================="

    # 1. Baseline
    echo "--- 1. Baseline (${SCENE}) ---"
    if python train.py \
        --source_path "${BASE_DATASET_DIR}/${SCENE}" \
        --model_path "${DIR_BASE}/${SCENE}" \
        --vggt_mode full_pipeline \
        --vggt_ply_name "vggt_omega_colmap.ply" \
        --eval \
        --wandb_name "vggt_colmapBase_${SCENE}"; then
        echo "✅ Finished Baseline for ${SCENE}"
    else
        echo "❌ Error in Baseline for ${SCENE}"
    fi

    # 2. No Pruning & No Densification
    echo "--- 2. No DP (${SCENE}) ---"
    if python train.py \
        --source_path "${BASE_DATASET_DIR}/${SCENE}" \
        --model_path "${DIR_NODP}/${SCENE}" \
        --vggt_mode full_pipeline \
        --vggt_ply_name "vggt_omega_colmap.ply" \
        --eval \
        --start_pruning 30000 \
        --densify_from_iter 30000 \
        --wandb_name "vggt_colmap_noDP_${SCENE}"; then
        echo "✅ Finished No DP for ${SCENE}"
    else
        echo "❌ Error in No DP for ${SCENE}"
    fi

    # 3. No Pruning
    echo "--- 3. No Pruning (${SCENE}) ---"
    if python train.py \
        --source_path "${BASE_DATASET_DIR}/${SCENE}" \
        --model_path "${DIR_NOPRUNE}/${SCENE}" \
        --vggt_mode full_pipeline \
        --vggt_ply_name "vggt_omega_colmap.ply" \
        --eval \
        --start_pruning 30000 \
        --wandb_name "vggt_colmap_noPrune_${SCENE}"; then
        echo "✅ Finished No Pruning for ${SCENE}"
    else
        echo "❌ Error in No Pruning for ${SCENE}"
    fi

    # 4. No Densification
    echo "--- 4. No Densification (${SCENE}) ---"
    if python train.py \
        --source_path "${BASE_DATASET_DIR}/${SCENE}" \
        --model_path "${DIR_NODENSIFY}/${SCENE}" \
        --vggt_mode full_pipeline \
        --vggt_ply_name "vggt_omega_colmap.ply" \
        --eval \
        --densify_from_iter 30000 \
        --wandb_name "vggt_colmap_noDensify_${SCENE}"; then
        echo "✅ Finished No Densification for ${SCENE}"
    else
        echo "❌ Error in No Densification for ${SCENE}"
    fi

done

echo "🎉 All scenes and all configurations completed!"
