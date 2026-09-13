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

echo "Reinstalling effrdel to include missing python modules..."
uv pip install ./submodules/effrdel --no-build-isolation --reinstall

echo "=========================================================="
echo "ENVIRONMENT IS FULLY INSTALLED - RUNNING VERIFICATION"
echo "=========================================================="

python -c "
import sys
try:
    import torch
    print(f'[OK] PyTorch version: {torch.__version__}')
    print(f'[OK] CUDA available: {torch.cuda.is_available()}')
    
    import mmcv
    print(f'[OK] MMCV version: {mmcv.__version__}')
    
    import diff_triangle_rasterization
    print('[OK] diff_triangle_rasterization imported successfully')
    
    import simple_knn
    print('[OK] simple_knn imported successfully')
    
    import rdel
    print('[OK] rdel imported successfully')
    
    print('\nSUCCESS! ALL MODULES ARE CORRECTLY INSTALLED AND FUNCTIONAL!')
except Exception as e:
    print('\n[ERROR] An import failed!')
    print(e)
    sys.exit(1)
"
