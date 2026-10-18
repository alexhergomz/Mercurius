# Mercurius

**Mercurius turns a trained hybrid language model into one that reads very long contexts on modest hardware.**
It rebuilds the memory-hungry parts of a model such as Qwen3.5 into cheaper ones, then teaches the rebuilt model
to behave like the original (or a larger teacher) by distillation, training directly against its deployed 4-bit form.
The first model, Mercurius-1-4B, reads a **1M-token context in about 5 GB** instead of about 37 GB.

<p align="center">
  <b>student &nbsp;→&nbsp; surgery &nbsp;→&nbsp; calibration &nbsp;→&nbsp; distillation + QAT &nbsp;→&nbsp; export</b><br>
  <i>one config file, one command, open data</i>
</p>

## Quick start

```bash
bash scripts/setup_spark.sh --teacher          # venv, PyTorch (CUDA 13), Mercurius, patched llama.cpp
bash scripts/teacher_server.sh models/Qwen3.5-35B-A3B-Q4_K_M.gguf &    # only for server teachers
mercurius run configs/qwen3.5-4b.yaml          # convert -> distill + QAT -> export (resumes after any stop)
mercurius eval runs/perihelion-4b/export --text data/wikitext.txt
```

Stages can also run one by one (`mercurius convert | train | export`), and every config value can be overridden on
the command line: `mercurius run configs/qwen3.5-9b.yaml train.lr=2e-5 qat.weights=iq4_nl`.

## What the pipeline does

| stage | what changes | cost |
|---|---|---|
| fold norms | RMSNorm gains move into the projections; folded norms become one learned scalar (ScaleNorm) | exact |
| GDN-2 lift | Gated DeltaNet → Gated DeltaNet-2: channel-wise decay, erase and write gates, row-tiled from the scalar gates; q/k heads expanded to the value heads | exact |
| absorbable MLA | each attention layer's K/V → one CARE latent (whitened SVD, ranks water-filled across layers in multiples of 128) + a small exact RoPE key (TransMLA RoRoPE); decode absorbs the up-projections | lossy, measured per layer |
| distillation | VeRA adapters (rank 1024) on every projection, dense latents, per-head query maps in the RoPE commutant; reverse KL + TAID + excess cross-entropy against one or more teachers | training |
| QAT | from `qat.start_step` the forward sees the deployed model: merged weights rounded to their format, the KV cache through its quantizer | training |
| export | adapters merged, weights packed in their formats, a self-contained checkpoint | exact (bit-identical to the QAT forward) |

### Configuration

One YAML file per run (`configs/`); every key and its default is in `src/mercurius/config.py`, and unknown keys
are an error. The main choices:

- **student**: any Hugging Face checkpoint of the Qwen3.5 / Qwen3-Next hybrid family (or plain-attention Qwen3);
  multimodal checkpoints are loaded text-only.
- **teachers**: a list. `kind: hf` loads a model in-process (`model: student` = the student's own original
  weights, i.e. self-distillation; the same checkpoint can be both); `kind: server` queries a llama.cpp server for
  final hidden states and reads the output head straight from the GGUF. Several teachers combine by weight, as a sum
  of divergences (`teacher_combine: loss`) or as one mixture target (`mixture`).
- **convert.mla**: latent budget, `rank_multiple`, number of RoPE components, low-rank key-RMS (`k_rms`).
- **train.amp**: `bf16` (default), `fp16` (with a gradient scaler) or `none`. Trainables always keep fp32 masters.
- **qat**: weight formats (`q4_0` default, `iq4_nl`, `q8_0`, `bf16`; `nf4` only to reproduce Mercurius-1),
  separate formats for the GDN-2 gates and the embedding, and the KV-cache format: `foveakv` (default: FoveaKV's
  gain-shape 4-bit pages with block means and a randomized Hadamard rotation), `turboquant`, `int` or `none`.

### Outputs

`runs/<name>/`: `converted/` (the model after surgery, loadable on its own), `ckpt/` (resume state and trained
tensors), `export/` (the deployable model), `config.resolved.json` and `train_log.json`. Any of the model
directories loads with `mercurius.arch.model.load_model(path)` -- no surgery is replayed and no side files are
needed.

## What v2 fixed

The first model was built with a research trainer (`src/mercurius/legacy/`, kept for reproducibility). Deploying
it (llama.cpp port, HgRT) produced [a list of issues](docs/porting_findings.md); v2 addresses them:

| finding | v2 |
|---|---|
| odd latent ranks (579, 451, ...) blocked block formats and tensor cores | ranks water-filled in multiples of `rank_multiple` (128) |
| NF4 weights: a format no standard runtime runs | QAT against standard formats (q4_0 / iq4_nl / q8_0), exported packed |
| GDN layers: 16 key heads shared by 32 value heads, expanded at run time anyway | q/k heads expanded to the value heads at conversion (exact) |
| the linear layers were called "KDA" and lifted again at load time | one `GatedDeltaNet2` class, saved as such |
| the decay gate carried two low-rank corrections (VeRA + an in-class LoRA) | one adapter per projection |
| a release was a research checkpoint needing `build()`, rotation files and cwd-relative paths | self-contained checkpoints (`load_model`) |
| RoPE key quantized before RoPE (un-rotation on every read) | the cache quantizer sees the post-RoPE key |
| key RMS needed a full extra projection | optional low-rank key-RMS (`convert.mla.k_rms`) |
| trainables in bf16 (small steps rounded away) | fp32 masters, mixed-precision compute (AMP) |
| teacher head dumped by hand from the GGUF | read from the GGUF automatically |

## Repository layout

```
src/mercurius/
  config.py      the run configuration (defaults, validation, overrides)
  cli.py         mercurius run | convert | train | export | eval | info
  arch/          the converted model: GDN-2, absorbable MLA / plain attention, norms, save / load
  convert/       exact stages (norm folding, GDN-2 lift), MLA conversion + sequential calibration, pipeline
  adapters.py    VeRA
  quant/         weight formats, KV-cache quantizers (FoveaKV, TurboQuant, int), QAT
  teachers.py    in-process and llama.cpp-server teachers
  train/         training mix, losses (divergences, TAID), trainer
  export.py      merged, packed, self-contained checkpoints
  data/          episode / math data builders and loaders
  legacy/        the Mercurius-1 research code (trainer, surgery variants, evaluations)
configs/         run configurations
scripts/         setup, teacher server, data builders (scripts/data)
research/        Mercurius-1 experiment scripts and logs
tests/           CPU tests on a tiny model: exactness of every exact stage, formats, end-to-end run, resume
docs/            decision log, findings, data policy, deployment
```

## Mercurius-1-4B at a glance

| | Original Qwen3.5-4B | Mercurius |
|---|---|---|
| KV cache per token | 32 KiB (bf16) | **2.3 KiB** (4-bit) — **13.7× smaller** |
| Cached values per token | 16,384 | 4,640 — 3.53× fewer before quantization |
| Weights | 8.4 GB (bf16) / 2.2 GB (4-bit) | **2.25 GB** (4-bit, trained for it) |
| Memory at 8k / 128k / 1M tokens (computed) | 2.5 / 6.5 / **36.6 GB** (4-bit weights, f16 KV) | 2.3 / 2.6 / **4.8 GB** |
| Measured in llama.cpp, 64k / 512k tokens | 4.5 / **19.3 GB** (f16 KV); 3.2 / 9.2 GB (4-bit KV) | 2.9 / **4.4 GB** |
| Generation speed at 128k (Jetson Orin) | 15.7 tok/s (f16 KV); 8.9 (4-bit KV) | **15.6 tok/s** |
| Prefill speed at 128k (Jetson Orin) | 555 tok/s (f16 KV) | 340 tok/s (1.6× slower) |
| Attention decode | standard | MLA, fully **absorbed** (no key rebuild per token) |
| 24 of 32 layers | Gated DeltaNet (linear) | **GDN-2** (channel-wise gates), exact lift |

Memory is for one sequence: weights + KV cache + the fixed 52 MB recurrent state. Measured rows: llama.cpp's
GPU buffers (also the attention mask and workspaces), IQ4_NL weights for both models; see
[docs/deployment.md](docs/deployment.md).

### Results

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

**Reasoning and code** (greedy, no thinking, 768 new tokens; paired per problem):

| | GSM8K | HumanEval | MBPP |
|---|---|---|---|
| Original Qwen3.5-4B (4-bit weights, 16-bit KV) | 91.7% | 78.0% | 69.6% |
| **Mercurius, deployed 4-bit** | 89.3% | 75.0% | 64.6% |
| difference vs original (p) | −2.4 (0.004) | −3.0 (0.51) | −5.1 (0.09) |
| Previous run, full precision | 88.8% | 75.6% | 65.4% |

The deployed model keeps most of the original's reasoning and code ability, with a
small loss that matches the 4-bit cost above.

**GSM8K is in-domain.** The math training data includes GSM8K's *train* split and
OpenMathInstruct (built from GSM8K / MATH train problems). The *test* split used here
is disjoint and was decontaminated (13-gram), so there is no leakage, but the score is
flattered relative to models without that data. Compare it only with the original
(which was also trained on such data) and with the previous run (same data).
HumanEval and MBPP have no in-domain training data.

Details and every measurement: [docs/decisions.md](docs/decisions.md) (#64–#68).

## Data

Only openly licensed data that allows commercial use, with no share-alike and no
privacy issues ([docs/data_policy.md](docs/data_policy.md)):

| share of tokens | source |
|---|---|
| 70.6% | FineWeb-Edu, long documents (≥ 8k tokens) |
| 19.7% | agentic software-engineering episodes (permissive repositories only) |
| 9.7% | math reasoning (GSM8K *train* split, OpenMathInstruct text, procedural problems) |

Everything is decontaminated against every evaluation set (13-gram overlap).

## Documentation

- [docs/porting_findings.md](docs/porting_findings.md) -- issues found deploying Mercurius-1 and how v2 answers them.
- [docs/decisions.md](docs/decisions.md) -- every design decision of Mercurius-1 with the measurement behind it.
- [docs/data_policy.md](docs/data_policy.md) -- licences, provenance and filters.
- [docs/deployment.md](docs/deployment.md) -- the llama.cpp port of Mercurius-1.
- [docs/evaluation.md](docs/evaluation.md), [docs/roadmap.md](docs/roadmap.md).

## Hardware

Developed on one NVIDIA GB10 (DGX Spark, 128 GB unified memory); deployment tested on a Jetson AGX Orin 64 GB.
The trainer caps its CUDA allocator (`train.guards.mem_cap_gb`), stops cleanly when free memory runs low and
resumes from its last save, and pauses when the GPU runs hot.

## Acknowledgements

Built on Qwen3.5 (Apache-2.0), flash-linear-attention, llama.cpp, and the ideas of TransMLA, DeepSeek-V2 MLA,
CARE / SVD-LLM, Gated DeltaNet-2, Kimi Delta Attention, VeRA, TAID, TurboQuant and FoveaKV.
