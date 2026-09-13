# Mercurius

Turning a trained Qwen3.5 into a cheaper-to-serve architecture — and adapting it
back to health — entirely on a single Jetson AGX Orin.

Nothing here is trained from scratch. The pipeline edits the architecture of
already-trained weights (linear-attention lift, positional-encoding removal,
KV-cache compression), then recovers the damage by distilling from the
unmodified original. Every surgery stage runs in high precision, before any
quantization.

Mercurius is the Roman god of translation and of boundaries — of carrying
something across into a new form, and of doing it quickly. Sizes are parameter
counts; the family name is the only mythological one.

## Result

Full pipeline versus the unmodified model it was carved out of, after 150 steps
(1.23 M tokens) of live-teacher distillation:

| model | ppl@2048 | ppl@8192 | retr@4k | retr@16k |
|---|---|---|---|---|
| Qwen3.5-0.8B (original) | 12.801 | 18.285 | 13.487 | 12.689 |
| + transKDA + NoPE | 12.501 | 17.428 | 13.811 | 13.539 |
| **+ transKDA + NoPE + 4× KV** | **12.581** | **17.440** | 13.684 | 12.823 |

**−1.72% ppl@2048 and −4.62% ppl@8192 at a 4× smaller KV cache**, with retrieval
costing +1.46% @4k and +1.05% @16k.

The caveat belongs on the same page as the number: both converted arms have seen
1.23 M tokens of FineWeb-Edu distillation and the original has seen none, so
"better than the teacher" partly reflects that extra adaptation. The defensible
claim is *parity-or-better at 4× smaller KV after 0.047 of an epoch* — not that
surgery improves a model.

Measured against a matched control that was trained identically but left
uncompressed, the cost of 4× KV compression falls from **+17.2%** ppl@8192
before recovery to **+0.05%** after.

## The pipeline

Each stage runs on the output of the last. Stages that claim exactness are
verified against a matched perturbation control — absolute thresholds are
useless here, since a 1e-7 perturbation moves logits by 5e-3.

| stage | what it does | gate |
|---|---|---|
| **A** `surgery/norm_fusion` | folds each RMSNorm gain into its consumer projections, leaving a parameter-free norm | bit-exact |
| **B** `surgery/kda_lift` | lifts GatedDeltaNet to Kimi Delta Attention by tiling the per-head scalar decay into a diagonal | bit-exact at init |
| **C** `surgery/rope_dial` | drives rotary frequencies to identity (NoPE); reversible, so a sweep costs only forward passes | +12.70% ppl@2048 |
| **D** `surgery/transmla` | replaces K and V with a shared low-rank latent, factorized in the **whitened** basis | 4.00× KV |
| **R** `recovery/train` | distils from the live original with exact full-vocabulary KL, computed in chunks | — |

Stage D's whitening is worth up to **26.76 pp** of perplexity against plain SVD
at the same rank, which is why `convert_to_mla` refuses to run without
calibration covariances unless explicitly overridden.

## Quick start

```bash
pip install -e .

# fetch the base model and corpora
python scripts/get_data.py

# stages A+B -> ckpt/qwen3.5-0.8b-stageAB
python scripts/build_stage_ab.py

# calibration covariances for Stage D's whitening
python scripts/save_covs.py

# stages C+D + recovery
python -m mercurius.recovery.train \
    --dial nope --seed-decay --seed-alpha 0 \
    --mla-dc 256 --mla-covs cache/kv_covs.pt \
    --live-teacher --seq 8192 \
    --grad-checkpoint --ckpt-above 8192 \
    --train-attn --train-norms \
    --steps 150 --eval-every 50 --lr 3e-5
```

Runs are resumable — weights, optimizer moments, schedule position and both
sampler RNG streams. Restarting from a weight checkpoint alone re-initializes
the moments and restarts the schedule at zero, which is a different run, not a
continuation.

## Layout

```
src/mercurius/
  paths.py          every filesystem path, overridable by env var
  models/           the KDA layer, quantization
  surgery/          stages A-D
  adapters/         LoRA, LayerScale, SSMax
  calibration/      activation covariances (plain and attention-weighted)
  recovery/         distillation training, layer-local transfer, logit cache
  eval/             perplexity, retrieval, long-context, generation
experiments/        ablations and one-off diagnostics
scripts/            data fetch, stage builders, disk guard
docs/               findings.md, recipe.html
```

`docs/findings.md` is the running record of what has been **measured**, what the
literature says, and what is still unproven — organized by confidence, with a
deliberate section for claims this project made confidently and then disproved.
Read it before trusting anything here.

## Hardware notes

Built for a Jetson AGX Orin 64 GB (sm_87, unified memory), which shapes several
non-obvious choices:

- **`causal_conv1d` must be built from source** (`TORCH_CUDA_ARCH_LIST=8.7`) —
  no published wheel targets this board. Worth 12× on that op, which runs 18
  times per forward.
- **No `torch.compile`.** Inductor decomposes SDPA instead of preserving
  FlashAttention, materializing a 16 GiB n² attention matrix at 32k context. It
  regresses 13% at short context and OOMs at long.
- **Norm gains are held in fp32.** At `w ≈ 0.5` the bf16 ULP is 3.91e-3 while an
  Adam step at 2e-4 is ~2e-4, so the update rounds to exactly zero.
- **Gradient checkpointing is length-conditional** (`--ckpt-above`): worth 1.43×,
  and mandatory above 8192 where the un-checkpointed peak no longer fits.
- **Disk is guarded.** The root filesystem is the only filesystem, and writes
  refuse below 8 GB free rather than discovering a full disk mid-save.

## Status

No run in this project has reached convergence — every one ended on wall-clock
with the loss still descending. The vision tower is intact and untouched at the
original checkpoint but is dropped at load; reattachment is unwritten.
