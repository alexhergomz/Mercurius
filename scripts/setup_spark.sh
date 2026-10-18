#!/usr/bin/env bash
# One-time setup on an NVIDIA DGX Spark (GB10, aarch64, CUDA 13) or any CUDA Linux box.
#   bash scripts/setup_spark.sh            # Python env + Mercurius
#   bash scripts/setup_spark.sh --teacher  # also build the patched llama.cpp used as a teacher server
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
# PyTorch with CUDA 13 (aarch64 + x86_64 wheels; GB10 is sm_121). Override TORCH_INDEX for other CUDA versions.
pip install torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
pip install -e ".[kernels,teachers,monitor,test]"
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
try:
    import fla; print("fla", fla.__version__)
except Exception as e:
    print("fla missing (PyTorch reference GDN-2 path will be used, slowly):", e)
PY
if [ "${1:-}" = "--teacher" ]; then
  LLAMA=${LLAMA_DIR:-third_party/llama.cpp}
  [ -d "$LLAMA" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
  # long prompts as per-token hidden states in n_ubatch chunks (LLAMA_SPLIT_EMBD_NONE=1)
  git -C "$LLAMA" apply --check "$PWD/patches/llama.cpp-split-pooling-none-embeddings.patch" 2>/dev/null \
    && git -C "$LLAMA" apply "$PWD/patches/llama.cpp-split-pooling-none-embeddings.patch" \
    || echo "patch did not apply cleanly to this llama.cpp revision; it was made against 4ceb171"
  cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA/build" -j --target llama-server
fi
echo "ready: . .venv/bin/activate && pytest -q && mercurius run configs/<config>.yaml"
