# Deployment: running Mercurius-1-4B

The released model needs this repository's `build()` to rebuild it (see the README). For inference without the
training code there is a llama.cpp port with ready-made GGUF files.

## llama.cpp (GGUF)

- **Fork:** [alexhergomz/llama-mercurius](https://github.com/alexhergomz/llama-mercurius), branch `mercurius`
  (llama.cpp → TheTom/llama-cpp-turboquant → this fork). Build, usage and supported hardware: its `docs/mercurius.md`.
- **Files:** the `gguf/` folder of the Hugging Face repository: exact NF4 (recommended; the trained 4-bit weights bit
  for bit), IQ4_NL, Q4_K, Q5_K and f16. Stock llama.cpp, Ollama and LM Studio cannot load them.

```bash
git clone -b mercurius https://github.com/alexhergomz/llama-mercurius && cd llama-mercurius
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-server
./build/bin/llama-server -m mercurius-1-4b-NF4.gguf -ngl 99 -c 131072 -fa on
```

**Hardware:** NVIDIA CUDA, Ampere and newer (tested on Jetson AGX Orin; other GPUs not yet tested); Turing compiles
but is not run-tested; CPU works but is slow. Not supported yet: AMD ROCm, Apple Metal, Vulkan, SYCL.

### What the port keeps

Everything that makes the model small: GDN-2 with channel-wise decay / erase / write gates (shared VeRA adapters,
factored per-head bases), absorbed MLA with the decoupled RoPE key, the TurboQuant 4-bit cache exactly as trained
(rotation + Lloyd-Max codebook + fp16 norms; ~2.4 KB per token over the 8 attention layers), and NF4 weights as a new
GGUF type.

Attention reads the packed cache directly:
- **Decode** (and short batches): absorbed attention, split-KV flash decoding straight from the 4-bit codes (one
  block per range of 64-token chunks, running softmax, RoPE keys rebuilt on tensor cores).
- **Prefill** (`-fa on`): the expanded form (keys 448 wide, values 256), which costs ~1.7x less per query–key pair
  than the absorbed form once ~140 queries share the expansion. The cache is expanded and attended in 16k-token
  slices with flash attention in partial mode, merged by log-sum-exp, so temporary memory is fixed (~150 MB) at any
  context length and no attention matrix or full expanded cache is ever stored.

### Measured (Jetson AGX Orin 64 GB, MODE_50W)

llama-server, `-fa on -b 2048 -ub 512`, IQ4_NL weights for both models, against Qwen3.5-4B (same 24 + 8 hybrid
layout). Memory = llama.cpp's GPU buffers (weights, cache, recurrent state, compute).

| Context | Mercurius: memory / prefill / generation | Qwen3.5-4B, f16 KV | Qwen3.5-4B, 4-bit (turbo4) KV |
|---|---|---|---|
| 64k | 2.86 GB / 143 s / 19.0 tok/s | 4.49 GB / 97 s / 20.3 | 3.21 GB / 104 s / 13.3 |
| 128k | 3.08 GB / 384 s / 15.6 | 6.60 GB / 236 s / 15.7 | 4.06 GB / 265 s / 8.9 |
| 256k | 3.52 GB / 1163 s / 11.4 | 10.8 GB / 630 s / 11.1 | 5.76 GB / 737 s / 5.4 |
| 512k | **4.40 GB** / 3852 s / 7.15 | 19.3 GB / 1891 s / 7.02 | 9.15 GB / 2325 s / 3.11 |

- Memory grows ~4–8x more slowly than Qwen3.5-4B's; the README's computed figures hold in a real runtime.
- Generation matches Qwen3.5-4B with an f16 cache (which needs 2–4x the memory) and is 1.4–2.3x faster than with a
  4-bit cache.
- Prefill is 1.5–2x slower: 448-wide keys mean ~1.4x the work per pair, the 448-wide flash-attention kernel runs at
  ~2/3 of the 256-wide one's throughput (single-stage pipeline, Q kept in shared memory), and the cache is re-expanded
  for every 512-token batch (a 2048 batch recovers ~11%). A dedicated kernel for this width is the next step.

### Verification

- Perplexity: NF4 GGUF 3.390 vs the PyTorch deployed model 3.392 on the same 12k tokens; IQ4_NL 3.432, Q4_K 3.422,
  Q5_K 3.407, f16 3.390.
- Long context: 16k-token slices vs one slice, 32k-token perplexity 5.5367 vs 5.5366.
- Greedy decoding NF4 vs f16 GGUF, 5 prompts x 32 tokens: 4 identical, 1 diverges at a near-tied token.
- Every new op is tested against its CPU implementation (`test-backend-ops`).

Issues found while porting (odd latent ranks, NF4 as a format, cache layout, config fields): [porting_findings.md](porting_findings.md).
