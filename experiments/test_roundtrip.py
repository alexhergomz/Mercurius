"""Save/load round trip for the converted checkpoint (standalone)."""
import sys, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.models.kda import convert_to_kda, load_kda_model, Qwen3_5KDAGatedDeltaNet
from mercurius.paths import BASE_MODEL, STAGE_AB

MODEL = str(BASE_MODEL)
OUT = str(STAGE_AB)
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


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)

    print("building stage A+B model ...", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, device_map="cuda").eval()
    fuse_model(m, verbose=False)
    convert_to_kda(m, m.config, lora_rank=0, verbose=True)
    before = logits_for(m, batch)

    # save in bf16: fp32 is needed DURING surgery, not for storage
    print("\nsaving (bf16) ...", flush=True)
    m.to(torch.bfloat16).save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    del m; torch.cuda.empty_cache()
    size = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT)) / 2**30
    print(f"  checkpoint: {size:.2f} GiB")
    print(f"  config kda_lift: "
          f"{getattr(AutoConfig.from_pretrained(OUT), 'kda_lift', None)}")

    print("\nloading back with load_kda_model ...", flush=True)
    m2 = load_kda_model(OUT, dtype=torch.float32)
    la = get_trunk(m2).layers[0].linear_attn
    print(f"  layer0 type    : {type(la).__name__}")
    print(f"  in_proj_a      : {tuple(la.in_proj_a.weight.shape)}")
    print(f"  A_log          : {tuple(la.A_log.shape)}")
    after = logits_for(m2, batch)

    r = rel(before, after)
    print(f"\n  round-trip relL2 vs pre-save: {r:.3e}")
    print(f"  (bf16 storage floor ~1e-3; exact would be 0 in fp32 storage)")
    print(f"\n  ROUND TRIP: {'PASS' if r < 5e-3 else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
