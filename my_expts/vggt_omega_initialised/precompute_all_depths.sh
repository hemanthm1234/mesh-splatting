#!/bin/bash
set -e

# Ensure we are in the root of the mesh-splatting repository
cd /data1/hemanth/mesh-splatting

BASE_DIR="/data1/hemanth/datasets/tandt"
SCENES=("Barn" "Caterpillar" "Church" "Ignatius" "Meeting_room" "Truck")

echo "================================================================="
echo "Precomputing VGGT depth scales for all scenes (16 workers)"
echo "================================================================="

for SCENE in "${SCENES[@]}"; do
    SCENE_PATH="$BASE_DIR/$SCENE"
    DEPTHS_DIR="$SCENE_PATH/depth"
    
    echo ">> Generating Depth Scaling Parameters for $SCENE..."
    
    python utils/make_depth_scale_vggt.py \
        --base_dir "$SCENE_PATH" \
        --depths_dir "$DEPTHS_DIR"
        
    echo "✅ $SCENE done."
    echo "-----------------------------------------------------------------"
done

echo "🎉 All depth scales precomputed successfully!"
