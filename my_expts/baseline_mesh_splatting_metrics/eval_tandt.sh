#!/bin/bash
set -e

EXP_DIR="/data1/hemanth/mesh-splatting/my_expts/baseline_mesh_splatting_metrics"
OUTPUT_DIR="$EXP_DIR/output/truck"
NUM_ITERATIONS=${1:-30000}
MESH_PATH="$OUTPUT_DIR/train/ours_${NUM_ITERATIONS}/fuse_unbounded.ply"

echo "Evaluating extracted mesh..."

if [ ! -f "$MESH_PATH" ]; then
    echo "Warning: Mesh file not found at $MESH_PATH. Did mesh.py complete successfully?"
    exit 1
fi

FILE_SIZE_MB=$(du -m "$MESH_PATH" | cut -f1)
echo "Mesh file size: ${FILE_SIZE_MB}MB"

# Python script to extract basic mesh statistics (vertices, faces)
python3 -c "
import trimesh
import json
import os

mesh_path = '$MESH_PATH'
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
    print(f'Mesh stats saved to {out_path}')
    print(f'Vertices: {stats[\"num_vertices\"]}, Faces: {stats[\"num_faces\"]}')
else:
    print('Mesh file not found!')
"
