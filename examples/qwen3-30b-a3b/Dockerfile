ARG BASE_IMAGE=rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

ARG MEGATRON_COMMIT=a1b00d4259e92dc4a07a0be2c24088fe827f4b6e
ARG TE_COMMIT=b2a0450804c0a999c42628834290f07d58d671ef

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ninja-build pybind11-dev \
        libopenmpi-dev libibverbs-dev rdma-core && \
    rm -rf /var/lib/apt/lists/* && \
    pip install pybind11 packaging ninja cmake psutil

# Build ROCm Transformer Engine for MI350X from source.
RUN git clone https://github.com/ROCm/TransformerEngine.git /workspace/TransformerEngine && \
    cd /workspace/TransformerEngine && \
    git checkout "${TE_COMMIT}" && \
    git submodule update --init --recursive
RUN cd /workspace/TransformerEngine && \
    NVTE_USE_ROCM=1 \
    NVTE_FRAMEWORK=pytorch \
    NVTE_ROCM_ARCH=gfx950 \
    CMAKE_ARGS="-DUSE_FUSED_ATTN_CK=ON -DCMAKE_HIP_ARCHITECTURES=gfx950" \
    MAX_JOBS=16 \
    pip install . --no-build-isolation

# Install the exact ROCm Megatron revision used by the local qwen3 example.
RUN git clone https://github.com/ROCm/Megatron-LM.git /workspace/Megatron-LM && \
    cd /workspace/Megatron-LM && \
    git checkout "${MEGATRON_COMMIT}" && \
    pip install -e ".[training]" --no-build-isolation && \
    make -C megatron/core/datasets

WORKDIR /workspace/Lumen
COPY . /workspace/Lumen

# Build SonicMoE-enabled AITER and install Lumen.
RUN cd /workspace/Lumen/third_party/aiter && \
    MAX_JOBS=16 pip install -e . --no-build-isolation --no-deps && \
    python -c "from aiter.jit.core import get_module; m = get_module('module_hipb_groupedgemm'); assert hasattr(m, 'hipb_grouped_mm'); assert hasattr(m, 'hipb_multistream_mm')"
RUN cd /workspace/Lumen && \
    pip install -e . --no-build-isolation --no-deps && \
    pip install \
        pytest pytest-xdist \
        "huggingface_hub>=0.24" "datasets>=2.16" "transformers>=4.43" \
        setuptools-scm "numpy>=1.26" && \
    pip install "peft>=0.9" --no-deps

# HipKittens is excluded because this benchmark is BF16.
RUN python -c "import pathlib, transformer_engine as te; root = pathlib.Path(te.__file__).parent; data = b''.join(p.read_bytes() for p in root.rglob('*.so')); assert b'ck_grouped' in data, 'CK grouped GEMM symbols missing'"

ENV PYTHONPATH="/workspace/Lumen:/workspace/Lumen/third_party/aiter:/workspace/Megatron-LM"
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV GPU_MAX_HW_QUEUES=8
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV USE_HIPBLASLT=1
ENV TORCH_BLAS_PREFER_HIPBLASLT=1

WORKDIR /workspace/Lumen/examples/qwen3-30b-a3b
