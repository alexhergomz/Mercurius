"""Combined gate for Stages A + B, with perturbation controls.

Absolute thresholds are meaningless on this architecture: a 1e-7 relative weight
perturbation amplifies to a ~5e-3 max logit delta through 24 layers of recurrent
state. So every exactness claim is measured against random-noise controls of
known-harmless magnitude, plus a negative control that should clearly fail.
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.surgery.kda_lift import install_kda_kernel, restore_gdn_kernel, verify_all_layers
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


def load():
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32,
                                             device_map="cuda")
    m.eval()
    return m


def report(tag, ref, new):
    d = (ref - new).abs()
    mx = d.max().item()
    rl = ((ref - new).norm() / ref.norm()).item()
    t1 = (ref.argmax(-1) == new.argmax(-1)).float().mean().item() * 100
    print(f"  {tag:<38} max {mx:.3e}   relL2 {rl:.3e}   top1 {t1:7.3f}%", flush=True)
    return mx, rl


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(TEXTS, return_tensors="pt", padding=True, truncation=True,
                max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}

    print("reference (unmodified) ...", flush=True)
    ref_model = load()
    ref = logits_for(ref_model, batch)
    print(f"  logits {tuple(ref.shape)}  |mean| {ref.abs().mean():.4f}\n", flush=True)

    # ---------- B2: parameter lift is exact ----------
    print("=== B2: lifted gate params reproduce scalar decay ===", flush=True)
    n, before, after, worst = verify_all_layers(ref_model)
    print(f"  GDN layers lifted            : {n}")
    print(f"  gate params  {before:,} -> {after:,}  (+{(after-before)/1e6:.2f} M dense)")
    print(f"  worst |g_diag - g_scalar|    : {worst:.3e}")
    # exact copies of a row through a different-shaped GEMM still differ by
    # fp32 reduction-order rounding; ~1e-6 on g of magnitude ~1e-1 is that.
    print(f"  {'EXACT' if worst < 1e-5 else 'DEVIATION EXCEEDS GEMM ROUNDING'}"
          f" (fp32 GEMM reduction-order floor)\n", flush=True)
    del ref_model; torch.cuda.empty_cache()

    # ---------- controls ----------
    print("=== controls: harmless perturbations ===", flush=True)
    ctrl = {}
    for eps in (1e-7, 1e-6):
        m = load()
        g = torch.Generator(device="cuda").manual_seed(1)
        for lin in fused_targets(m):
            w = lin.weight.data
            w.mul_(1.0 + eps * torch.randn(w.shape, generator=g, device=w.device,
                                           dtype=w.dtype))
        ctrl[eps] = report(f"random rel. noise {eps:.0e}", ref, logits_for(m, batch))
        del m; torch.cuda.empty_cache()

    # ---------- implementation-swap control ----------
    # Stage B changes WHICH ALGORITHM computes the same math, so the weight-
    # perturbation controls above are the wrong reference for it. Measure how
    # far a same-math implementation swap alone moves the logits: replace the
    # chunked GDN kernel with the recurrent GDN kernel. Identical mathematics,
    # different accumulation order -- exactly the class of difference B incurs.
    print("\n=== control: same-math implementation swap (GDN chunk -> GDN recurrent) ===",
          flush=True)
    import transformers.models.qwen3_5.modeling_qwen3_5 as qm
    _chunk, _recur = qm.torch_chunk_gated_delta_rule, qm.torch_recurrent_gated_delta_rule
    qm.torch_chunk_gated_delta_rule = _recur
    m = load()
    impl_ctrl = report("GDN chunked -> GDN recurrent", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()
    qm.torch_chunk_gated_delta_rule = _chunk

    # ---------- A alone ----------
    print("\n=== stage A alone (norm fusion) ===", flush=True)
    m = load(); fuse_model(m, verbose=False)
    a_res = report("A: fusion", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()

    # ---------- B1 alone ----------
    print("\n=== stage B alone (KDA kernel, broadcast decay) ===", flush=True)
    orig = install_kda_kernel()
    m = load()
    b_res = report("B: GDN -> KDA kernel", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()

    # ---------- A + B ----------
    print("\n=== stages A + B combined ===", flush=True)
    m = load(); fuse_model(m, verbose=False)
    ab_res = report("A+B", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()
    restore_gdn_kernel(orig)

    # ---------- negative control ----------
    print("\n=== negative control (should clearly fail) ===", flush=True)
    m = load()
    for lin in fused_targets(m):
        lin.weight.data.mul_(1.0 + 1e-3)
    neg = report("uniform +0.1% weight error", ref, logits_for(m, batch))
    del m; torch.cuda.empty_cache()

    # ---------- verdict ----------
    # Each stage is judged against the control that matches what it changes:
    #   A changes weights          -> weight-perturbation control
    #   B changes the algorithm    -> implementation-swap control
    bound_w = ctrl[1e-6][1] * 1.5
    bound_i = impl_ctrl[1] * 1.5
    print("\n=== VERDICT ===")
    print(f"  weight-perturbation bound (1.5x 1e-6 noise)  : {bound_w:.3e}")
    print(f"  implementation-swap bound (1.5x chunk->recur): {bound_i:.3e}")
    print()
    checks = (("A",   a_res,  bound_w, "weight perturbation"),
              ("B",   b_res,  bound_i, "implementation swap"),
              ("A+B", ab_res, max(bound_w, bound_i), "both"))
    allok = True
    for name, res, bound, which in checks:
        ok = res[1] <= bound
        allok &= ok
        print(f"  {name:<4} relL2 {res[1]:.3e}  vs {bound:.3e} ({which:<19}) "
              f"{'PASS' if ok else 'FAIL'}")
    negok = neg[1] > max(bound_w, bound_i)
    print(f"\n  negative control relL2 {neg[1]:.3e}  "
          f"({'correctly exceeds every bound' if negok else 'PROBLEM: did not fail'})")
    print(f"\n  GATE A+B: {'PASS' if allok and negok else 'FAIL'}")
    return 0 if (allok and negok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
