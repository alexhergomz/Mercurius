# Mercurius

**Mercurius turns a trained small language model into one that can read very long
contexts on modest hardware.** It takes Qwen3.5-4B, rebuilds its memory-hungry parts
into cheaper ones, and then teaches the rebuilt model to behave like the original by
distillation from a larger teacher. The result reads a **1M-token context in about
5 GB of memory** instead of about 37 GB, with 4-bit weights and a 4-bit KV cache.

<p align="center">
  <b>Qwen3.5-4B &nbsp;→&nbsp; surgery &nbsp;→&nbsp; calibration &nbsp;→&nbsp; distillation &nbsp;→&nbsp; 4-bit QAT</b><br>
  <i>one GPU box, open data, no training from scratch</i>
</p>

---

## At a glance

| | Original Qwen3.5-4B | Mercurius |
|---|---|---|
| KV cache per token | 32 KiB (bf16) | **2.3 KiB** (4-bit) — **13.7× smaller** |
| Cached values per token | 16,384 | 4,640 — 3.53× fewer before quantization |
| Weights | 8.4 GB (bf16) / 2.2 GB (4-bit) | **2.25 GB** (4-bit, trained for it) |
| Memory at 8k / 128k / 1M tokens | 2.5 / 6.5 / **36.6 GB** (4-bit weights, f16 KV) | 2.3 / 2.6 / **4.8 GB** |
| Attention decode | standard | MLA, fully **absorbed** (no key rebuild per token) |
| 24 of 32 layers | Gated DeltaNet (linear) | **GDN-2** (channel-wise gates), exact lift |

Memory is for one sequence: weights + KV cache + the fixed 52 MB recurrent state.

## How it works

Qwen3.5-4B is a hybrid: 24 linear-attention layers (Gated DeltaNet) and 8 softmax
attention layers. Only the 8 attention layers keep a cache that grows with context,
so that is where the compression goes.

```
 8 attention layers   K,V per head ──► one shared latent (CARE whitened SVD, water-filled ranks)
                      RoPE part    ──► one small exact RoPE key per layer (TransMLA RoRoPE)
                      → absorbable MLA: the cache holds the latent, not the keys

24 linear layers      GatedDeltaNet ──► GDN-2: per-channel decay, erase and write gates
                      lifted exactly from the trained weights, then adapted

whole model           distilled from Qwen3.5-35B-A3B (reverse KL + TAID + CE)
                      then quantization-aware training: NF4 weights, TurboQuant 4-bit KV
```

1. **Surgery.** Each change starts exactly where the original model was, or with a
   measured, small loss. The attention layers become Multi-head Latent Attention:
   keys and values share one latent per token, factorized in the *whitened* basis of
   real activations (CARE), with ranks water-filled across layers under a fixed
   budget. Rotary position encoding is split out first (TransMLA's RoRoPE: one
   orthogonal mix of heads per frequency, which commutes with RoPE), so decoding is
   fully absorbed like DeepSeek's MLA.
2. **Calibration.** Covariances come from the training mix itself, layer by layer, on
   the partly compressed model (sequential calibration).
3. **Distillation.** A live 35B-A3B teacher (llama.cpp server) gives hidden states;
   the student matches its distribution with reverse KL, a TAID schedule, and
   cross-entropy. Adapters are VeRA (rank 1024) plus the dense latents. Long-context
   windows (up to 32k tokens) are part of the mix.
4. **Quantization-aware training.** The model trains against its deployed form:
   NF4 weights with adapters merged, and a TurboQuant-MSE 4-bit KV cache (fixed random
   rotation + Lloyd-Max codebook + one norm per token). The rotations and codebooks
   are saved with the model.

## Results

Run `v3-absorb`: 3,900 steps (25M tokens), the last 750 with quantization-aware
training. Two versions are reported: the full-precision model just before QAT
(step 3,150) and the **final deployed model** (4-bit weights, 4-bit KV cache, 4-bit
embedding).

**Long-context retrieval** (RULER, 5 multi-needle tasks, 32k tokens):

| | exact match | value NLL ↓ |
|---|---|---|
| Original Qwen3.5-4B | 100% | 0.062 |
| Mercurius, full precision (step 3,150) | **90.5%** | **0.081** |
| **Mercurius, deployed 4-bit** | 79.5% | 0.098 |
| Previous run, full precision (before the RoPE fix) | 73.5% | 0.126 |

At 64k the deployed 4-bit model beats the previous full-precision run on all three
multi-key tasks (e.g. hardest task NLL 0.100 vs 0.433); the original still leads.

**Long-document modelling** (PG-19, 12 books, NLL above the original; lower is better):

| position | 0–2k | 2–8k | 8–16k | 16–32k | 32–64k |
|---|---|---|---|---|---|
| Mercurius, full precision | **−0.007** | +0.003 | +0.009 | +0.008 | +0.015 |
| Mercurius, deployed 4-bit | +0.026 | +0.034 | +0.040 | +0.039 | +0.048 |
| Previous run, full precision | +0.003 | +0.012 | +0.015 | +0.019 | +0.034 |

**What 4 bits cost.** About +0.03 nats, nearly the same at every position, so it is a
global cost, not a long-range one. The 4-bit KV cache (TurboQuant) and embedding cost
only ~0.009 of it; the rest is the 4-bit weights. The QAT phase was short (750 steps,
learning rate already decaying) and recovered about 30%. Next run: start QAT much
earlier, or keep the most sensitive weights at 8 bits.

Reasoning and code benchmarks (GSM8K, HumanEval, MBPP) of the previous run were level
with or above a short-trained baseline; the final model was not re-run on them yet.
Details and every measurement: [docs/decisions.md](docs/decisions.md) (#64–#68).

## Quick start

```bash
pip install -e .

# Teacher: llama.cpp server with the split-embedding patch (long prompts in 8k chunks)
git -C <llama.cpp> apply patches/llama.cpp-split-pooling-none-embeddings.patch   # commit 4ceb171
LLAMA_SPLIT_EMBD_NONE=1 llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf --embeddings \
  --pooling none --embd-normalize -1 -ngl 99 -c 67584 -b 8192 -ub 8192 \
  --parallel 2 --cache-ram 0 --port 8077

# Train (conversion, calibration, distillation and QAT in one run)
bash experiments/run_v2.sh --go
bash experiments/run_v2.sh --go --resume     # after any interruption

# Evaluate a snapshot (rebuilds the exact trained model, then RULER / long PPL)
python experiments/replay_check.py --arm A=ckpt/adapters-<tag>-step<N>.pt \
  --covs cache/mla_seqcal_<key>_covs.pt --groups cache/mla_seqcal_<key>_groups.json \
  --dial c0 --merge-eval --expect <CE@2048> <CE@8192>
python -m mercurius.eval.ruler --arms A=<ckpt> --quantize --merge-eval --fast-infer \
  --covs <covs> --mla-groups <groups> --dial c0 --lengths 32768 65536
```

Runs resume exactly: weights, optimizer moments, LR schedule, TAID state, every
sampler and RNG stream, and a fingerprint of the data files are saved every 25 steps.

## Data

Only openly licensed data that allows commercial use, with no share-alike and no
privacy issues ([docs/data_policy.md](docs/data_policy.md)):

| share of tokens | source |
|---|---|
| 70.6% | FineWeb-Edu, long documents (≥ 8k tokens) |
| 19.7% | agentic software-engineering episodes (permissive repositories only) |
| 9.7% | math reasoning (GSM8K, OpenMathInstruct text, procedural problems) |

Everything is decontaminated against every evaluation set (13-gram overlap).

## Repository layout

```
src/mercurius/
  surgery/       MLA conversion (transmla.py, mla_rope.py), GDN-2 / KDA lifts,
                 per-head query maps, RoPE dial, norm fusion
  calibration/   activation covariances, sequential MLA calibration
  models/        GDN-2 / KDA layers, QAT (qat.py), fast inference paths
  recovery/      distillation trainer, teacher-server client
  eval/          RULER, long-context perplexity, retrieval, benchmarks
experiments/     run scripts, evaluations, diagnostics
patches/         llama.cpp patch for long-prompt hidden states
docs/            decision log, data policy, evaluation notes, roadmap
```

## Documentation

- [docs/decisions.md](docs/decisions.md) — every design decision with the measurement
  behind it, including the mistakes and how they were found.
- [docs/data_policy.md](docs/data_policy.md) — licences, provenance and filters.
- [docs/evaluation.md](docs/evaluation.md) — how results are measured.
- [docs/roadmap.md](docs/roadmap.md) — what comes next.

## Hardware

Developed on one NVIDIA GB10 (128 GB unified memory). The trainer and the teacher
share that memory, so the trainer stops cleanly when free memory runs low and resumes
from its last save.

## Status

Research project. Next: a longer QAT phase (to close the 4-bit gap), GSM8K /
HumanEval / MBPP on the final model, and a fused absorbed-decode kernel for
deployment.

## Acknowledgements

Built on Qwen3.5 (Apache-2.0), flash-linear-attention, llama.cpp, and the ideas of
TransMLA, DeepSeek-V2 MLA, CARE / SVD-LLM, Gated DeltaNet-2, Kimi Delta Attention,
VeRA, TAID and TurboQuant.
