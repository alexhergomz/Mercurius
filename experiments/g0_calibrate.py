"""Calibrate G0's threshold instead of guessing it.

Question: is the fused model's 1.19e-2 max logit delta a real error, or is it
fp32 rounding amplified through 24 layers of a recurrent architecture?

Control: perturb the ORIGINAL weights by relative noise of the same magnitude
fp32 rounding produces (~1e-7), change nothing else, and measure the resulting
logit delta. If a provably harmless perturbation moves the logits as much as
the fusion does, the fusion is numerically indistinguishable from exact.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.paths import BASE_MODEL

MODEL = str(BASE_MODEL)
TEXTS = [
    "The Jetson AGX Orin has unified memory shared between CPU and GPU.",
    "In linear attention, the delta rule updates a fixed-size state matrix.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a",
    "Cuando el modelo pierde la codificacion posicional, las capas lineales",
]


@torch.no_grad()
def logits_for(model, batch):
    return model(**batch).logits.float().clone()


def fused_targets(model):
    """The exact weight tensors Stage A touches."""
    out = []
    for layer in get_trunk(model).layers:
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            out += [la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a]
        else:
            sa = layer.self_attn
            out += [sa.q_proj, sa.k_proj, sa.v_proj]
        out += [layer.mlp.gate_proj, layer.mlp.up_proj]
    return out


def report(tag, ref, new):
    d = (ref - new).abs()
    print(f"  {tag:<34} max {d.max().item():.3e}   relL2 "
          f"{((ref-new).norm()/ref.norm()).item():.3e}   top1 "
          f"{(ref.argmax(-1)==new.argmax(-1)).float().mean().item()*100:.4f}%",
          flush=True)
    return d.max().item()


def load():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             device_map="cuda")
    m.eval()
    return m


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(TEXTS, return_tensors="pt", padding=True, truncation=True,
                max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}

    print("loading reference ...", flush=True)
    model = load()
    ref = logits_for(model, batch)
    print(f"  logits {tuple(ref.shape)}  |mean| {ref.abs().mean():.4f}\n", flush=True)

    print("=== CONTROLS: harmless perturbations of the same magnitude ===", flush=True)
    for eps in (1e-7, 1e-6):
        m = load()
        g = torch.Generator(device="cuda").manual_seed(1)
        for lin in fused_targets(m):
            w = lin.weight.data
            w.mul_(1.0 + eps * torch.randn(w.shape, generator=g,
                                           device=w.device, dtype=w.dtype))
        report(f"random rel. noise {eps:.0e}", ref, logits_for(m, batch))
        del m; torch.cuda.empty_cache()

    print("\n=== STAGE A FUSION ===", flush=True)
    m = load()
    fuse_model(m, verbose=False)
    fused_max = report("norm fusion (should be exact)", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()

    print("\n=== NEGATIVE CONTROL: a genuinely wrong fusion ===", flush=True)
    m = load()
    for lin in fused_targets(m):
        lin.weight.data.mul_(1.0 + 1e-3)      # 0.1% systematic error
    report("uniform +0.1% weight error", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()

    print(f"\nCONCLUSION: fusion max delta {fused_max:.3e}; compare against the")
    print("controls above. If it sits between the 1e-7 and 1e-6 noise rows, the")
    print("fusion is numerically exact and the threshold should be set from the")
    print("architecture's amplification factor, not an arbitrary 1e-3.")


if __name__ == "__main__":
    main()
