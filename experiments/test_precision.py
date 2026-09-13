"""Separate loader correctness from storage precision.

1. fp32 save/load round trip -> should be ~0 if the loader is correct.
2. bf16 rounding in memory (no save) -> isolates what bf16 storage costs.

Matters beyond this test: Stage A's fusion produces genuine fp32 weights
(W * (1+gamma)). If bf16 cannot hold them, NF4 -- far coarser -- is a much
bigger problem, and the fuse-then-quantize ordering needs scale migration.
"""
import sys, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.models.kda import convert_to_kda, load_kda_model
from mercurius.paths import BASE_MODEL

MODEL = str(BASE_MODEL)
OUT32 = "ckpt/tmp-fp32"
TEXTS = [
    "The Jetson AGX Orin has unified memory shared between CPU and GPU.",
    "In linear attention, the delta rule updates a fixed-size state matrix.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a",
    "Cuando el modelo pierde la codificacion posicional, las capas lineales",
]


@torch.no_grad()
def logits_for(m, b):
    return m(**b).logits.float().clone()


def rel(a, b):
    return ((a - b).norm() / a.norm()).item()


def build(fuse=True, kda=True):
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             device_map="cuda").eval()
    if fuse:
        fuse_model(m, verbose=False)
    if kda:
        convert_to_kda(m, m.config, lora_rank=0, verbose=False)
    return m


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}

    # --- is bf16 lossy for the UNFUSED model? (it shipped as bf16, so: no) ---
    print("=== bf16 rounding, UNFUSED model (baseline) ===", flush=True)
    m = build(fuse=False, kda=False)
    ref_plain = logits_for(m, batch)
    for p in m.parameters():
        p.data = p.data.to(torch.bfloat16).to(torch.float32)
    print(f"  relL2 after bf16 round: {rel(ref_plain, logits_for(m, batch)):.3e}"
          f"   (expect ~0: the checkpoint already shipped bf16)")
    del m; torch.cuda.empty_cache()

    # --- is bf16 lossy for the FUSED model? ---
    print("\n=== bf16 rounding, FUSED model ===", flush=True)
    m = build(fuse=True, kda=True)
    ref_fused = logits_for(m, batch)
    for p in m.parameters():
        p.data = p.data.to(torch.bfloat16).to(torch.float32)
    r_bf16 = rel(ref_fused, logits_for(m, batch))
    print(f"  relL2 after bf16 round: {r_bf16:.3e}")
    print(f"  -> fusion creates weights bf16 cannot hold" if r_bf16 > 1e-2
          else "  -> fused weights survive bf16")
    del m; torch.cuda.empty_cache()

    # --- fp32 save/load round trip: validates the loader itself ---
    print("\n=== fp32 save/load round trip (loader correctness) ===", flush=True)
    if os.path.isdir(OUT32):
        shutil.rmtree(OUT32)
    m = build()
    before = logits_for(m, batch)
    m.save_pretrained(OUT32, safe_serialization=True)
    tok.save_pretrained(OUT32)
    del m; torch.cuda.empty_cache()

    m2 = load_kda_model(OUT32, dtype=torch.float32)
    after = logits_for(m2, batch)
    r32 = rel(before, after)
    print(f"  relL2: {r32:.3e}")
    print(f"  LOADER: {'CORRECT' if r32 < 1e-6 else 'BUG — investigate'}")
    del m2; torch.cuda.empty_cache()
    shutil.rmtree(OUT32, ignore_errors=True)

    print("\n=== conclusion ===")
    if r32 < 1e-6 and r_bf16 > 1e-2:
        print("  loader is correct; the 1.17e-1 round trip was bf16 STORAGE.")
        print("  Store converted checkpoints in fp32, and expect NF4 to need")
        print("  activation-aware scale migration after fusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
