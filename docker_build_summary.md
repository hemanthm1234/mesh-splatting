# Mesh-Splatting Docker Build Troubleshooting & Solutions

This document logs all the errors encountered while trying to build the Docker image for `mesh-splatting`, along with the exact solutions applied to fix them.

## Current Status
The build is currently running and is at the final step (`Step 14/15`), compiling the custom C++ extensions (`diff-triangle-mesh-rasterization`, `simple-knn`, `effrdel`) for 6 different CUDA architectures. Because this is a very heavy compilation task (effectively compiling the extensions 18 times in total), it is taking around 30-45 minutes. It is expected to finish successfully as there have been no errors.

---

## Issues Encountered and Resolved

### 1. `ModuleNotFoundError: No module named 'pkg_resources'`
- **Error:** When attempting to compile `mmcv` from source, the build failed immediately because it couldn't find `pkg_resources`.
- **Cause:** The base `uv pip install` step installed the latest version of `setuptools` (v70+). Starting with v70, `setuptools` completely removed the `pkg_resources` module, which older libraries like `mmcv` still rely on for their `setup.py`.
- **Solution:** Modified the `Dockerfile` to strictly reinstall `setuptools==69.5.1` right before running the `mmcv` build command.

### 2. `ImportError: undefined symbol: ncclCommResume`
- **Error:** During the compilation of submodules (`compile.sh`), the script tried to run `import torch`, which crashed with an undefined NCCL symbol error.
- **Cause:** To match the initial CUDA 13.2 container, PyTorch was instructed to download from the `cu132` index. However, PyTorch doesn't have a stable release for CUDA 13.2 yet. It downloaded an experimental nightly build that had a broken NVIDIA communication library (`libnccl`).
- **Solution:** Modified the `Dockerfile` to pull the stable PyTorch `cu124` (CUDA 12.4) wheel instead. CUDA is perfectly backward compatible, so this wheel runs flawlessly on CUDA 13.x drivers.

### 3. Conflicting Local Requirements (Unsatisfiable Requirements)
- **Error:** The Docker build failed at the `requirements-frozen.txt` installation step with "requirements are unsatisfiable".
- **Cause:** Because we switched to the stable PyTorch `cu124` wheel, PyTorch now required stable dependencies (like `triton==3.1.0` and `nvidia-cublas-cu12==12.4.5.8`). However, the `requirements-frozen.txt` generated from your local machine had rigidly pinned `triton==3.7.1` and newer `nvidia-*` libraries (which belonged to the broken PyTorch nightly). 
- **Solution:** Added explicit filters to `build_docker.sh` (`grep -v "^triton=="` and `grep -v "^nvidia-"`) to strip these specific dependencies out of the frozen list, allowing PyTorch to dynamically pull its own correct, stable dependencies.

### 4. `RuntimeError: The detected CUDA version (13.2) mismatches...`
- **Error:** The C++ extensions for `simple-knn` and `effrdel` refused to compile, throwing a major version mismatch error.
- **Cause:** PyTorch's `cpp_extension.py` has a hardcoded security check that forcefully prevents compiling C++ extensions if the major version of CUDA in the system (13.2) does not perfectly match the major version PyTorch was compiled with (12.4).
- **Solution:** Downgraded the base Docker container image from `nvidia/cuda:13.2.0-devel-ubuntu22.04` to `nvidia/cuda:12.4.1-devel-ubuntu22.04`. This perfectly harmonized the container's CUDA version with PyTorch's CUDA version.

### 5. `IndexError: list index out of range` (GPU Detection Failure)
- **Error:** PyTorch's compiler crashed while trying to parse the `TORCH_CUDA_ARCH_LIST`.
- **Cause:** Docker build environments run completely isolated from the host machine's GPUs. Because no GPU was detected during the build, PyTorch's compiler generated an empty list of architectures to build for, and crashed when trying to access it.
- **Solution:** Explicitly set `ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"` in the `Dockerfile`. This bypasses GPU auto-detection and instructs the compiler to build binaries for all modern GPUs, exactly matching the requested list (A100, RTX 3090, Ada Lovelace, H100, Blackwell).

### 6. Accidental Nightly Fetches via `unsafe-best-match`
- **Error:** Occasionally, `uv` would still pull PyTorch 2.6.x nightly builds from PyPI because of the `--index-strategy unsafe-best-match` flag when we removed the strict version constraint from the frozen requirements.
- **Solution:** Hard-pinned PyTorch directly in the `Dockerfile` (`RUN uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124`). This guarantees a stable build environment regardless of index strategies.
