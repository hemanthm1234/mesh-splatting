#!/bin/bash
set -e

# Base directory
BASE_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"truck"}
DATASET_DIR="/data1/hemanth/datasets/tandt/$SCENE"
DEPTH_ANYTHING_DIR="/data1/hemanth/Depth-Anything-V2"

if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory $DATASET_DIR not found. Please run download_tandt.sh first."
    exit 1
fi

echo "Setting up Depth Anything V2..."

cd /data1/hemanth

if [ ! -d "$DEPTH_ANYTHING_DIR" ]; then
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DEPTH_ANYTHING_DIR"
fi

cd "$DEPTH_ANYTHING_DIR"
mkdir -p checkpoints

if [ ! -f "checkpoints/depth_anything_v2_vitl.pth" ]; then
    echo "Downloading Depth Anything V2 ViT-L weights..."
    wget -c https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth -P checkpoints/
fi

echo "Generating depth maps for truck scene..."

# Output directory for depth maps
OUT_DIR="$DATASET_DIR/depth"
mkdir -p "$OUT_DIR"

# Check if depth_params.json already exists to skip inference
if [ -f "$DATASET_DIR/sparse/0/depth_params.json" ]; then
    echo "Depth maps and depth_params.json already exist! Skipping inference..."
else
    # Ensure we have the right environment (this assumes the active env has torch/torchvision/cv2 etc.)
    # Run Depth Anything V2 inference
    python run.py \
        --encoder vitl \
        --pred-only \
        --grayscale \
        --img-path "$DATASET_DIR/images" \
        --outdir "$OUT_DIR"
fi

echo "Generating depth_params.json..."
cd /data1/hemanth/mesh-splatting

python utils/make_depth_scale.py \
    --base_dir "$DATASET_DIR" \
    --depths_dir "$OUT_DIR"

echo "Depth map generation complete for $DATASET_DIR!"
