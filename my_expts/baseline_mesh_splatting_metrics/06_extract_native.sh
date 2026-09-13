#!/bin/bash
set -e
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting
EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
SCENE=${1:-"truck"}
OUTPUT_DIR="$EXP_DIR/output/${SCENE}_baseline"

# The checkpoint iteration (passed from run_experiment, or default 2000)
NUM_ITERATIONS=${2:-2000}
CHECKPOINT_DIR="$OUTPUT_DIR/point_cloud/iteration_${NUM_ITERATIONS}"

echo "=== [Step 5] Extracting Native Mesh (No TSDF) ==="
cd /data1/hemanth/mesh-splatting

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint $CHECKPOINT_DIR not found. Did training finish?"
    exit 1
fi

python create_ply.py "$CHECKPOINT_DIR" --out "$OUTPUT_DIR/native_mesh_${NUM_ITERATIONS}.ply"

echo "Native mesh exported to $OUTPUT_DIR/native_mesh_${NUM_ITERATIONS}.ply"

# Calculate basic file size and counts
FILE_SIZE_MB=$(du -m "$OUTPUT_DIR/native_mesh_${NUM_ITERATIONS}.ply" | cut -f1)
echo "Mesh file size: ${FILE_SIZE_MB}MB"

python3 -c "
import trimesh
import json
import os

mesh_path = '$OUTPUT_DIR/native_mesh_${NUM_ITERATIONS}.ply'
out_path = os.path.join('$OUTPUT_DIR', 'mesh_stats.json')

if os.path.exists(mesh_path):
    mesh = trimesh.load(mesh_path)
    stats = {
        'num_vertices': len(mesh.vertices),
        'num_faces': len(mesh.faces),
        'file_size_mb': $FILE_SIZE_MB
    }
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f'Vertices: {stats[\"num_vertices\"]}, Faces: {stats[\"num_faces\"]}')
else:
    print('Failed to load exported PLY for stats.')
"
echo "=== Native Mesh Extraction Complete! ==="
