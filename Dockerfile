FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install system dependencies & Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl ninja-build build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# 2. Set up a lightning-fast uv virtual environment
RUN pip install uv
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Set CUDA architectures to build for (so PyTorch doesn't crash trying to detect a GPU during docker build)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# 3. Pre-install crucial build dependencies using standard pip to guarantee environment sanity
# We use get-pip.py to guarantee pip is installed inside the venv (Ubuntu's python-venv often omits it).
# This ensures setuptools is installed properly and provides pkg_resources.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    python -m pip install setuptools==69.5.1 wheel ninja packaging pybind11 uv

# 4. Install PyTorch FIRST because mmcv and other extensions need it to build!
RUN uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu124

# 5. Build mmcv completely before processing the rest of the requirements.
# We build it manually from source to completely bypass PEP-517 pyproject hooks which hide pkg_resources.
# We also reinstall setuptools==69.5.1 right before building, because the PyTorch install step might have upgraded it to 70+ which removes pkg_resources.
RUN git clone --branch v2.2.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    cd /tmp/mmcv && \
    python -m pip install setuptools==69.5.1 wheel ninja packaging pybind11 uv && \
    MMCV_WITH_OPS=1 python setup.py bdist_wheel && \
    python -m pip install dist/*.whl && \
    rm -rf /tmp/mmcv

# 6. Install exactly matched frozen requirements
COPY requirements-frozen.txt .
RUN uv pip install -r requirements-frozen.txt --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match --no-build-isolation

# 5. Copy ONLY the submodules, compile them into /opt/venv, and then delete them!
COPY submodules/ ./submodules/
COPY compile.sh .
RUN bash compile.sh && \
    uv pip install ./submodules/simple-knn --no-build-isolation && \
    uv pip install ./submodules/effrdel --no-build-isolation

# 6. Clean up the build files so the image contains NOTHING but the static environment
RUN rm -rf submodules compile.sh requirements-frozen.txt

# The /app directory is now completely empty and ready for your network mount!
CMD ["/bin/bash"]
