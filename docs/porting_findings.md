# Porting findings: Mercurius-1-4B in deployment (PyTorch deployed form, llama.cpp / GGUF)

Written while deploying the final model outside the training code: a PyTorch deployed form (NF4 weights, the packed
TurboQuant cache as a real runtime) and the llama.cpp port ([alexhergomz/llama-mercurius](https://github.com/alexhergomz/llama-mercurius),
see [deployment.md](deployment.md)). Each item says what was seen, why it matters, and what a fix would look like.
"Inherited" = comes from Qwen3.5-4B as is. These are inputs for the next training run, not bugs in the released model.

## Architecture / shapes

1. **GDN layers: 16 key heads vs 32 value heads (inherited GVA), but GDN-2 makes it pointless.** Qwen3.5 shares each
   q/k head between two value heads. GDN-2's decay `g`, erase gate `b` and write gate `w` are per *value* head and
   channel (32 x 128), so every kernel (fla `chunk_gdn2`, llama.cpp `gated_delta_net`) needs q/k expanded to 32 heads
   anyway: fla's gdn2 asserts `q.shape == k.shape == g.shape`, and our PyTorch path does `repeat_interleave(2)`. The
   sharing only saves q/k projection parameters while costing a q/k copy per layer at run time.
   Ideal: equal key and value head counts (32/32), or gates per key head.

2. **Attention head geometry is asymmetric and non-standard: QK 448 wide, V 256.** The expanded keys are
   64 (RoPE) + 192 (shared residual) + 192 (per group). Stock flash-attention kernels stop at 256, so PyTorch fell
   back to a ~6x slower kernel; we wrote a Triton kernel for it. In the absorbed form QK is 64 + r (up to 720) and V is r.

3. **Latent widths are odd per layer: 579, 451, 390, 427, 579, 543, 471, 656.** None is a multiple of 16/32/64, so
   - `attn_v_up` (r columns) cannot be stored in any block-quantized type (block 32/64) and stays f16;
   - bitsandbytes warns "inner dimension (579) is not aligned for fast kernel ... falling back to slower
     implementation" on every up-projection;
   - kernels need tail handling everywhere.
   Fix without retraining: zero-pad the up-projections to a multiple of 64 (exact); the TurboQuant codes stay width r.
   Better: choose ranks as multiples of 64 when compressing.

4. **The per-group key normaliser needs a full extra projection.** `k_rms` is a 2560 -> 1024 (D x G) projection per
   attention layer whose only use is 4 scalars per token (the RMS of each group's original key). It costs ~11 MB of
   NF4 weights and a matmul per token per attention layer, for 4 numbers.

5. **The RoPE key is quantized before RoPE, in a rotated basis.** The cache stores TurboQuant codes of `R0 k_rope`;
   every read decodes, un-rotates with `R0^T` (64 x 64 per cached token) and then applies RoPE. The un-rotation can't
   be folded into the query because RoPE sits between them. Works, but it is per-token-per-read work that a
   "rope then quantize" layout would not need (that would change the quantization realisation vs training).

6. **`num_key_value_heads = 4` and `head_dim = 256` in config.json describe the expanded form only.** In the absorbed
   form the latent is shared by all heads; the 4 "KV heads" survive only as the per-group nope up-projection, the
   per-group key RMS and the per-group V up-projection. Tools reading the config (memory estimators, servers)
   compute a stock GQA cache size (~90 KB/token) instead of the real ~2.8 KB/token.

## Weight format

14. **The weights are trained for NF4, which is not a standard deployment format.** NF4 (bitsandbytes'
    NormalFloat4: 16 fixed non-uniform levels, block 64, double-quantized fp8 block scales) exists in bitsandbytes /
    PyTorch, but not in llama.cpp/GGUF, TensorRT-LLM, vLLM's fast kernels, MLX or ONNX runtimes. To ship it we had to
    add a new GGUF tensor type (`NF4`, block 64 + fp32 scale) and CUDA/CPU kernels to the llama.cpp fork; the 4-bit
    types every runtime supports (Q4_0/Q4_K/IQ4_NL, INT4 AWQ/GPTQ) re-round the already-quantized weights and lose
    accuracy (perplexity 3.390 for NF4 vs 3.432 / 3.422 for IQ4_NL / Q4_K on the same text). NF4 levels are also irregular
    floats, so the int8 dot-product tricks the standard kernels use don't apply directly: the first NF4
    matrix-vector kernel ran at ~60% of IQ4_NL's speed (an int16 table kernel later brought it to ~85%). Training the QAT against a standard grid (e.g. IQ4_NL's
    levels with fp16 block-32 scales, or plain INT4 with group scales) would keep the QAT benefit and run everywhere.

## Cache format

7. **Each cached token stores its position (int32) and 4 fp32 key RMS values per attention layer.** That is 20 of
   ~300 bytes per row. The position is the cell index in a contiguous cache (redundant); the RMS values would be fine
   in fp16 (-8 bytes per row, ~3%).

8. **4-bit KV amplifies tiny arithmetic differences.** Equivalent computation orders (bf16 matmul order, fused vs
   unfused attention, CPU vs GPU) give 3-5% relative L2 differences in hidden states, and greedy decoding diverges
   at near-tied tokens between backends (PyTorch vs llama.cpp: 3-4 of 5 prompts identical for 32 tokens). Expected
   for a quantized cache, but worth stating when comparing implementations.

15. **The latent ranks are odd and irregular (579, 451, 390, 427, 579, 543, 471, 656).** Inner dimensions that are
    not multiples of 8 keep cuBLAS off its tensor-core paths: the prefill up-projection GEMMs ran at ~5 TFLOP/s until
    the weights were zero-padded to the decoded latent's width (multiple of 16) at run time (~2 MB per call;
    64k-cell slice set: 29 -> 10 ms per layer). Multiples of 64 would have cost nothing to train.

## Checkpoint / packaging

9. **The release is a research checkpoint, not a model.** Using it needs Mercurius's own `build()`: base Qwen3.5-4B ->
   stage A+B conversion (KDA/GDN-2 lift, covariances, groups), VeRA adapters (rank 1024, 11 patterns over 256
   modules), TurboQuant rotations from a separate file (regenerated if missing), cwd-relative paths, and a
   transformers 5.6-specific KDA forward (5.17 needed a port).

10. **Naming: the code calls the linear layers "KDA" (`Qwen3_5KDAGatedDeltaNet`); the deployed layers are GDN-2**
    (a "GDN-2 lift" replaces them at load). Easy to mis-identify the architecture from class names.

11. **NF4 of the embedding is not idempotent.** bitsandbytes double-quantizes the block scales, so re-quantizing the
    already-NF4 table changes codes; the exact table only comes from quantizing the original embedding once.
    Similarly the trained forward uses `w + (nf4(w) - w)` for `k_rope`, which is not bit-identical to `nf4(w)` and
    flips some 4-bit codes downstream.

12. **Unused feature in the checkpoint: the decay-tied rotary phase (`pe_c`) is implemented but off (None).**

13. **GDN-2 gates: three gates share one VeRA pair (A 1024 x 2560, B 4096 x 1024) on top of a tiled per-head base,
    plus a rank-32 LoRA on the decay gate only.** Efficient as stored (merging it would add 0.4-1 GB), but the decay
    gate has two different low-rank corrections (VeRA and LoRA), which looks like an artefact of stacking training stages.

## Port bugs found in the deployment code (fixed; listed so they are not mistaken for model issues)

- llama.cpp: VeRA A/B were placed in the CPU "input layer" -> 194 graph splits per decoded token (fixed: 2).
- llama.cpp: stacked-gate VeRA matmul took ggml's per-slice vec path (0.42 ms per layer) -> one GEMM.
- llama.cpp: decode attention unpacked the whole cache to fp32 twice per token per layer -> fused kernel reading the
  packed cache directly.
- llama.cpp: expanded prefill materialized the expanded K/V for the whole cache (5.6 KB per cached token, 2.9 GB at
  512k) -> one fused op processes the cache in 16k-cell slices (decode, expansion GEMMs, MMA flash attention in
  partial mode) merged online by log-sum-exp; ~150 MB of fixed workspace held in the op's compute-buffer slot.
  32k perplexity: single 64k slice 5.5366 vs 16k slices 5.5367.
- llama.cpp: decode attention wrote one partial result per 64-cell chunk (~39 MB per layer at 64k, more than the
  cache) and merged them with 16 blocks -> split-KV with a running softmax per block, cp.async staging, rope keys by a
  tensor-core GEMM, padded shared-memory strides: 15k cells 1322 -> 443 us per layer; at 64k decode 11.5 -> 18.3 tok/s
  (Qwen3.5-4B: 19.5 with f16 KV, 14.2 with turbo4 KV).
