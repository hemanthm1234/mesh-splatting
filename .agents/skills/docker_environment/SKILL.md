---
name: docker_environment
description: Comprehensive guidelines and troubleshooting steps for building Docker images from complex Python environments, specifically handling PyTorch, CUDA C++ extensions, and frozen dependencies.
---

# Building Complex PyTorch & CUDA Docker Environments

When tasked with converting a local Conda/Python environment (especially those involving heavy 3D/4D rendering, PyTorch, and custom CUDA C++ extensions) into a Docker image, agents **must** adhere to the following critical principles to avoid cascading compilation errors.

## 1. Freezing Requirements Safely
Never blindly `pip install -r requirements-frozen.txt` if the frozen list comes from a local machine with CUDA extensions.
- **Filter Out PyTorch & CUDA System Libraries:** Your build script must filter out `torch`, `torchvision`, `torchaudio`, `triton`, and `nvidia-*` packages from the frozen list.
- **Why:** Local frozen requirements are often rigidly pinned to specific nightly builds or versions that conflict when Docker resolves PyTorch from a specific index URL. Let PyTorch dynamically resolve its own stable `triton` and `nvidia-*` dependencies.
- **Example filter:**
  ```bash
  grep -ivE "^(torch|torchvision|torchaudio)==" requirements-frozen-raw.txt | \
  grep -v "^triton==" | \
  grep -v "^nvidia-" > requirements-frozen.txt
  ```

## 2. Pinning Stable PyTorch Versions
Do not rely on `--index-strategy unsafe-best-match` alone without pinning PyTorch.
- **Rule:** Explicitly pin PyTorch to a stable version in the `Dockerfile` (e.g., `torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1`).
- **Why:** Without explicit versions, package managers like `uv` may fetch experimental or nightly wheels (e.g., PyTorch 2.6 or `cu132` nightlies) which frequently contain broken symbols (like missing `ncclCommResume`).

## 3. The CUDA Major Version Mismatch Trap
- **Rule:** The major version of the Docker container's base image **MUST EXACTLY MATCH** the major version of CUDA that PyTorch was compiled with.
- **Example:** If you install PyTorch `cu124` (compiled with CUDA 12.4), your Dockerfile MUST start with `FROM nvidia/cuda:12.4.1-devel-ubuntu22.04`. It **CANNOT** be a `13.x` container.
- **Why:** PyTorch's `torch/utils/cpp_extension.py` contains a hardcoded security check. If the container's CUDA major version differs from PyTorch's, it will instantly throw a `RuntimeError` and refuse to compile any custom C++ extensions. 
- *Note:* Docker perfectly isolates CUDA versions, so it does not matter if the host machine uses CUDA 13.x; the container can safely use 12.x as long as the NVIDIA driver supports it.

## 4. The Docker GPU Detection Crash (`IndexError: list index out of range`)
- **Rule:** Always set `ENV TORCH_CUDA_ARCH_LIST` explicitly in the `Dockerfile` before compiling C++ extensions.
- **Example:** `ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"`
- **Why:** Docker build environments do not have a GPU attached during `docker build`. Without this environment variable, `cpp_extension.py` will try to query the non-existent GPU to determine the architecture, return an empty list, and crash with `IndexError: list index out of range`. Setting the variable forces the compiler to build fat binaries for the specified architectures.

## 5. The `setuptools` and `pkg_resources` Bug
- **Rule:** If the project compiles older libraries (like `mmcv` or custom C++ packages), ensure `setuptools` is strictly pinned to `<70`.
- **Example:** `RUN python -m pip install setuptools==69.5.1`
- **Why:** `setuptools` version 70.0.0 and above completely removed the `pkg_resources` module. Older `setup.py` scripts relying on it will immediately fail with `ModuleNotFoundError: No module named 'pkg_resources'`.

## Summary of the Ideal Flow
1. Start with an `nvidia/cuda` base image that perfectly matches your target PyTorch `cuXXX` wheel.
2. Install system dependencies, python, and pip/uv.
3. Explicitly install the exact, stable `torch` version from the correct `--extra-index-url`.
4. Set `ENV TORCH_CUDA_ARCH_LIST` to a comprehensive list of modern GPU architectures.
5. (Optional) Downgrade `setuptools` to `<70` if compiling older extensions.
6. Install custom extensions (like `mmcv`, `simple-knn`, etc.).
7. Install the filtered `requirements-frozen.txt`.
