#!/usr/bin/env bash
# Teacher server for `kind: server` teachers: final hidden states per token, raw (not L2-normalised).
#   bash scripts/teacher_server.sh models/Qwen3.5-35B-A3B-Q4_K_M.gguf [port] [context]
set -euo pipefail
GGUF=${1:?gguf path}; PORT=${2:-8077}; CTX=${3:-67584}
BIN=${LLAMA_SERVER:-third_party/llama.cpp/build/bin/llama-server}
LLAMA_SPLIT_EMBD_NONE=1 exec "$BIN" -m "$GGUF" --embeddings --pooling none --embd-normalize -1 \
  -ngl 99 -c "$CTX" -b 8192 -ub 8192 --parallel 2 --cache-ram 0 --port "$PORT"
