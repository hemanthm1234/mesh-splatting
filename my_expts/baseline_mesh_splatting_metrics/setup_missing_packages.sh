#!/bin/bash
set -e

echo "=== Setup Missing Environment Packages ==="
echo "Activating mesh_splatting environment..."
source ~/miniconda3/etc/profile.d/conda.sh || true
conda activate mesh_splatting

echo "Installing missing Python packages..."
# Install packages that were found to be missing during execution (including scikit-image for marching_cubes)
uv pip install open3d trimesh joblib wandb scikit-image

# Find the project root directory relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Downloading missing utils/mcube_utils.py..."
# The mesh-splatting codebase imports this file but forgot to include it in their repo
if [ ! -f "$PROJECT_ROOT/utils/mcube_utils.py" ]; then
    wget -qO "$PROJECT_ROOT/utils/mcube_utils.py" "https://raw.githubusercontent.com/hbb1/2d-gaussian-splatting/main/utils/mcube_utils.py"
fi

echo "Environment setup complete! All packages installed successfully."
