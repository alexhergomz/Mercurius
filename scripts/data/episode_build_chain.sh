#!/usr/bin/env bash
# Wait for the GPU to be free, serve the teacher (4-bit GGUF, q8 KV cache),
# build episodes to the token target, then stop the server.
cd "$(dirname "$0")/.."
export PATH=$PWD/.venv/bin:$PATH
while pgrep -f '^python -m mercurius\.(eval|recovery)' > /dev/null; do sleep 30; done
/home/alberto/llama.cpp/build/bin/llama-server \
  -m models/qwen3.5-27b-gguf/Qwen3.5-27B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 --ctx-size 1048576 --parallel 4 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja --alias teacher --no-webui \
  > logs/llama-server.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do sleep 10; curl -sf http://127.0.0.1:8080/health > /dev/null && break; done
curl -sf http://127.0.0.1:8080/health > /dev/null || { echo "server failed to start"; kill $SRV; exit 1; }
echo "teacher up (pid $SRV)"
python scripts/build_episodes.py --tasks 6 --workers 6 --read-lines 1000 --search-k 8 \
  --max-turns 60 --target-tokens "${TARGET:-4000000}" --out data/episodes/build1.jsonl \
  > logs/build-episodes.log 2>&1
rc=$?
kill $SRV; sleep 5
echo "build exit $rc"; tail -3 logs/build-episodes.log
