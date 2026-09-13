"""Produce the Stage A+B checkpoint: fused norms + KDA layers, LoRA-ready.

Saved in fp32. bf16 storage costs relL2 5.68e-3 on the fused model (measured),
and the plan keeps everything in high precision until the quantization step.
"""
import sys, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.models.kda import convert_to_kda, load_kda_model
from mercurius.paths import BASE_MODEL, STAGE_AB

MODEL = str(BASE_MODEL)
OUT = str(STAGE_AB)
LORA_RANK = 32
TEXTS = [
    "The Jetson AGX Orin has unified memory shared between CPU and GPU.",
    "In linear attention, the delta rule updates a fixed-size state matrix.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a",
    "Cuando el modelo pierde la codificacion posicional, las capas lineales",
]
CONTROL = 7.427e-04     # same-math implementation-swap bound


@torch.no_grad()
def logits_for(m, b):
    return m(**b).logits.float().clone()


def rel(a, b):
    return ((a - b).norm() / a.norm()).item()


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}

    print("reference ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             device_map="cuda").eval()
    ref = logits_for(m, batch)

    print("stage A: fusing norms ...", flush=True)
    fuse_model(m, verbose=False)
    print(f"stage B: lifting GDN -> KDA (lora rank {LORA_RANK}) ...", flush=True)
    convert_to_kda(m, m.config, lora_rank=LORA_RANK, verbose=True)

    r = rel(ref, logits_for(m, batch))
    print(f"\n  A+B relL2 vs original: {r:.3e}   bound {CONTROL*1.5:.3e}   "
          f"{'PASS' if r <= CONTROL * 1.5 else 'FAIL'}")
    if r > CONTROL * 1.5:
        raise SystemExit("refusing to save: conversion exceeded its bound")

    n_tr = sum(p.numel() for l in get_trunk(m).layers
               if hasattr(l, "linear_attn") and hasattr(l.linear_attn, "trainable_gate_parameters")
               for p in l.linear_attn.trainable_gate_parameters())
    print(f"  new trainable gate params: {n_tr/1e6:.2f} M across 18 layers")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    print(f"\nsaving fp32 -> {OUT}", flush=True)
    m.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    before = logits_for(m, batch)
    del m; torch.cuda.empty_cache()

    print("verifying reload ...", flush=True)
    m2 = load_kda_model(OUT, dtype=torch.float32)
    rr = rel(before, logits_for(m2, batch))
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 2**30
    print(f"  reload relL2: {rr:.3e}   {'EXACT' if rr == 0.0 else 'LOSSY'}")
    print(f"  checkpoint  : {size:.2f} GiB")
    print(f"\nSTAGE A+B CHECKPOINT READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
