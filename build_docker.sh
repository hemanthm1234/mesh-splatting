#!/bin/bash
set -e

echo "Initializing conda..."
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi

echo "Activating 'mesh_splatting' environment..."
conda activate mesh_splatting

echo "Freezing current environment to ensure EXACT match..."
uv pip freeze > requirements-frozen-raw.txt

echo "Filtering out local packages and custom CUDA modules so they can be built properly inside Docker..."
# We filter out the custom packages that are installed via local paths
# as well as any lines that contain "file://"
grep -v "diff-triangle-rasterization" requirements-frozen-raw.txt | \
grep -v "simple-knn" | \
grep -v "effrdel" | \
grep -v "file://" | \
grep -ivE "^(torch|torchvision|torchaudio)==" | \
grep -v "^triton==" | \
grep -v "^nvidia-" > requirements-frozen.txt

echo "Successfully created requirements-frozen.txt with exact versions."

echo "Building Docker image (this will take some time)..."
docker build -t mesh_splatting .

echo "Docker image built successfully! You can run it with:"
echo "docker run --gpus all -it mesh_splatting"
