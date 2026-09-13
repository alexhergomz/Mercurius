"""G0 — verify Stage A is function-preserving.

Two independent checks:

1. ALGEBRAIC: for each fused norm, confirm  (W @ diag(g)) @ x_hat == W @ (g * x_hat)
   on random inputs. Pure linear algebra, exact, catches arithmetic errors.

2. END-TO-END: run a fixed probe through the model before and after fusion and
   compare logits. Catches wiring errors the algebraic check cannot -- e.g.
   folding gamma into a projection that does not actually consume that norm.

Runs fp32 on CUDA. CPU is not an option: transformers dispatches Qwen3.5's
linear attention to fla's Triton kernels, which reject CPU pointers. Comparing
pre- vs post-fusion through the *same* deterministic code path isolates the
fusion's numerical error, so the chunked kernel's ~5e-3 cross-implementation
noise floor does not apply here.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.paths import BASE_MODEL

MODEL = str(BASE_MODEL)
PROBE_TEXTS = [
    "The Jetson AGX Orin has unified memory shared between CPU and GPU.",
    "In linear attention, the delta rule updates a fixed-size state matrix.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a",
    "Cuando el modelo pierde la codificacion posicional, las capas lineales",
]


def algebraic_check(dtype=torch.float32, n=2000, d=1024, out=3584, trials=20):
    """W' = W diag(g) applied to x_hat  ==  W applied to (g * x_hat)."""
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(trials):
        W = torch.randn(out, d, device="cuda", dtype=dtype)
        g = torch.randn(d, device="cuda", dtype=dtype).abs() + 0.05
        x = torch.randn(n, d, device="cuda", dtype=dtype)
        lhs = (x * g) @ W.T           # original: norm scales, then project
        rhs = x @ (W * g.unsqueeze(0)).T   # fused: gamma folded into W
        worst = max(worst, ((lhs - rhs).abs().max() /
                            lhs.abs().max().clamp_min(1e-9)).item())
    return worst


@torch.no_grad()
def logits_for(model, batch):
    return model(**batch).logits.float().clone()


def main():
    torch.manual_seed(0)
    # TF32 keeps only ~10 mantissa bits. Fusion changes W's actual values, so
    # W@(g*x) and (W@g)@x round differently at every layer -- that shows up as
    # a fusion "error" that is really just reduced-precision matmul.
    tf32 = "--tf32" in sys.argv
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    print(f"TF32 matmul: {'ENABLED' if tf32 else 'DISABLED'}\n", flush=True)

    print("=== check 1: algebraic identity ===", flush=True)
    for dt, name in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        w = algebraic_check(dt)
        print(f"  {name}: worst relative error {w:.3e}", flush=True)
    print("  (bf16 error is why surgery runs pre-quantization in fp32)\n", flush=True)

    print("=== check 2: end-to-end logits ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(PROBE_TEXTS, return_tensors="pt", padding=True,
                truncation=True, max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}
    print(f"  probe: {batch['input_ids'].shape[0]} seqs x "
          f"{batch['input_ids'].shape[1]} tokens", flush=True)

    print("  loading fp32 on cuda ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cuda")
    model.eval()

    print("  reference forward ...", flush=True)
    ref = logits_for(model, batch)
    print(f"    logits {tuple(ref.shape)}  |mean| {ref.abs().mean():.4f}", flush=True)

    # determinism control: same model, same input, twice
    ref2 = logits_for(model, batch)
    determinism = (ref - ref2).abs().max().item()
    print(f"    determinism control (same model twice): {determinism:.3e}", flush=True)

    print("\n  fusing ...", flush=True)
    fuse_model(model)

    print("\n  post-fusion forward ...", flush=True)
    new = logits_for(model, batch)

    delta = (ref - new).abs()
    rel = ((ref - new).norm() / ref.norm()).item()
    top1 = (ref.argmax(-1) == new.argmax(-1)).float().mean().item()

    print("\n=== G0 RESULT ===")
    print(f"  determinism floor    : {determinism:.3e}")
    print(f"  max abs logit delta  : {delta.max().item():.3e}")
    print(f"  mean abs logit delta : {delta.mean().item():.3e}")
    print(f"  relative L2          : {rel:.3e}")
    print(f"  top-1 agreement      : {top1*100:.4f}%")
    ok = delta.max().item() < 1e-3
    print(f"\n  VERDICT: {'PASS' if ok else 'FAIL'} (threshold 1e-3 max abs logit delta)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
