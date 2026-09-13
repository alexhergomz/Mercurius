# Mercurius

Mercurius converts a trained Qwen3.5 model into a cheaper architecture and then
repairs the damage by distillation. All steps run on one Jetson AGX Orin.

## Motivation

A trained model is expensive to produce and cheap to edit. Most of the cost of
serving it comes from parts that can be replaced: the KV cache grows with
context length, softmax attention is quadratic, and rotary position encoding
constrains length extrapolation.

Retraining from scratch with a better architecture is not possible on one
device. Editing the architecture of trained weights is. The edits below are
chosen so that each one either preserves the function exactly, or has a measured
cost that distillation can pay back.

The method needs no gradient budget beyond a short recovery run, and no data
beyond a general corpus, because the teacher is the original model.

## Method

Each stage runs on the output of the previous one, in high precision, before any
quantization. A stage that claims exactness is checked against a matched
perturbation control. Absolute thresholds do not work here: a 1e-7 perturbation
moves logits by 5e-3.

| stage | module | operation | cost |
|---|---|---|---|
| A | `surgery/norm_fusion` | fold each RMSNorm gain into the projections that consume it, leaving a norm with no parameters | exact |
| B | `surgery/kda_lift` | lift GatedDeltaNet to Kimi Delta Attention by tiling the per-head scalar decay into a diagonal | exact at init |
| C | `surgery/rope_dial` | set all rotary frequencies to identity (NoPE) | +12.70% ppl@2048 |
| D | `surgery/transmla` | replace K and V with one shared low-rank latent, factorized in the whitened basis | 4.00x smaller KV |
| R | `recovery/train` | distil from the unmodified model using exact full-vocabulary KL | — |

Stage C changes no weights and is reversible, so a sweep over how much rotary to
keep costs only forward passes.

Stage D factorizes in the whitened basis, which minimizes activation error
rather than weight error. This is worth up to 26.76 percentage points of
perplexity against plain SVD at the same rank, so `convert_to_mla` refuses to
run without calibration covariances unless told otherwise.

Stage R uses a live teacher rather than cached logits. A cache of top-k logits
is cheaper per step but wrong in two ways: truncation at k=64 over a 248,320
vocabulary closes only about 5% of the gap to full distillation, and a cache
built on short blocks conditions the teacher on less context than the student
sees, which makes the loss penalize correct long-range retrieval.

## Usage

```bash
pip install -e .

python scripts/get_data.py          # base model and corpora
python scripts/build_stage_ab.py    # stages A and B
python scripts/save_covs.py         # calibration covariances for stage D

python -m mercurius.recovery.train \
    --dial nope --seed-decay --seed-alpha 0 \
    --mla-dc 256 --mla-covs cache/kv_covs.pt \
    --live-teacher --seq 8192 \
    --grad-checkpoint --ckpt-above 8192 \
    --train-attn --train-norms \
    --steps 150 --eval-every 50 --lr 3e-5
```

Runs are resumable. A resume file holds weights, optimizer moments, schedule
position, and both sampler RNG streams. A weight checkpoint alone is not enough:
restarting from one resets the moments and restarts the schedule, which gives a
different run.

All paths resolve from one root. Set `MERCURIUS_ROOT` to relocate the tree, or
`MERCURIUS_CKPT`, `MERCURIUS_DATA`, `MERCURIUS_MODELS`, `MERCURIUS_CACHE`,
`MERCURIUS_LOGS` to move one directory.

## Layout

```
src/mercurius/
  paths.py          all filesystem paths, overridable by environment variable
  models/           the KDA layer, quantization
  surgery/          stages A to D
  adapters/         LoRA, LayerScale, SSMax
  calibration/      activation covariances, plain and attention-weighted
  recovery/         distillation training, layer-local transfer, logit cache
  eval/             perplexity, retrieval, long context, generation
experiments/        ablations and one-off diagnostics
scripts/            data fetch, stage builders, disk guard, sync from origin
docs/               findings.md, recipe.html
```

## Hardware notes

Built for a Jetson AGX Orin 64 GB (sm_87, unified memory). Several choices
follow from that:

- `causal_conv1d` must be built from source with `TORCH_CUDA_ARCH_LIST=8.7`. No
  published wheel targets this board. It gives 12x on that operation, which runs
  18 times per forward pass.
- `torch.compile` is not used. Inductor decomposes SDPA instead of keeping
  FlashAttention, which materializes a 16 GiB attention matrix at 32k context.
  It is 13% slower at short context and runs out of memory at long context.
- Norm gains are held in fp32. At `w = 0.5` the bf16 ULP is 3.91e-3 and an Adam
  step at 2e-4 is about 2e-4, so the update rounds to zero.
- Gradient checkpointing is conditional on sequence length (`--ckpt-above`). It
  costs 1.43x throughput and is required above 8192, where the peak without it
  no longer fits.
- Writes refuse below 8 GB free. The root filesystem is the only filesystem.

## Status

Work in progress. Nothing here is settled.

No run has reached convergence. Every run so far ended on wall-clock with the
loss still falling. The adaptation surface is under revision. The vision tower
is intact in the original checkpoint but is dropped at load; reattaching it is
not written.

`docs/findings.md` is the record of what has been measured, what the literature
says, and what is unproven. It is ordered by confidence and includes a section
for claims made in this project that were later disproved. Read it before
relying on anything here.
