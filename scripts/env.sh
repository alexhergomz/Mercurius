# source this before any work in this project
export PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="$PROJ/models/hf"
export HF_HUB_CACHE="$PROJ/models/hf/hub"
export TRITON_CACHE_DIR="$PROJ/env/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$PROJ/env/inductor-cache"
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST="8.7"
# Orin is unified memory; keep the allocator from fragmenting across the shared pool
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$PROJ/.venv/bin/activate"
