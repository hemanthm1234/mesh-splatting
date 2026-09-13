#!/bin/bash
set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"


SCENES=("Truck" "Meeting_room" "Church" "Caterpillar" "Ignatius")
BASE_DATASET_DIR="/data1/hemanth/datasets/tandt"

for SCENE in "${SCENES[@]}"; do
    DEPTH_PARAMS_FILE="${BASE_DATASET_DIR}/${SCENE}/vggt_omega/depth_params.json"
    if [ -f "${DEPTH_PARAMS_FILE}" ]; then
        echo "⏭️  Skipping ${SCENE}: ${DEPTH_PARAMS_FILE} already exists."
        continue
    fi

    echo "=================================================="
    echo "Precomputing VGGT Depth Scale for scene: ${SCENE}"
    echo "=================================================="
    python utils/make_depth_scale_vggt.py \
        --base_dir "${BASE_DATASET_DIR}/${SCENE}" \
        --depths_dir "${BASE_DATASET_DIR}/${SCENE}/depth"
done

echo "✅ All depth scale precomputations finished successfully!"
