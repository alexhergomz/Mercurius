"""Regression test for the KDA modeling class.

The monkeypatch verification established the target: Stage B moves the logits
by relL2 5.905e-04, against a same-math implementation-swap control at
7.427e-04. The class must reproduce that. If it doesn't, the class has a bug
the math didn't.

Also checks the three things the monkeypatch could not do:
  * lifted parameters exist as real nn.Parameters
  * they are trainable (gradients flow)
  * the conversion survives save_pretrained -> from_pretrained
"""
import sys, os, shutil, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mercurius.surgery.norm_fusion import fuse_model, get_trunk
from mercurius.models.kda import convert_to_kda, Qwen3_5KDAGatedDeltaNet
from mercurius.paths import BASE_MODEL, STAGE_AB

MODEL = str(BASE_MODEL)
OUT = str(STAGE_AB)
TEXTS = [
    "The Jetson AGX Orin has unified memory shared between CPU and GPU.",
    "In linear attention, the delta rule updates a fixed-size state matrix.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a",
    "Cuando el modelo pierde la codificacion posicional, las capas lineales",
]
TARGET_B = 5.905e-04       # from the verified monkeypatch
TARGET_AB = 6.228e-04
CONTROL = 7.427e-04        # same-math implementation swap


@torch.no_grad()
def logits_for(model, batch):
    return model(**batch).logits.float().clone()


def load(path=MODEL):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                             device_map="cuda")
    m.eval()
    return m


def rel(ref, new):
    return ((ref - new).norm() / ref.norm()).item()


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    batch = tok(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=48)
    batch = {k: v.cuda() for k, v in batch.items()}

    print("reference ...", flush=True)
    m = load()
    ref = logits_for(m, batch)
    del m; torch.cuda.empty_cache()

    cfg = AutoConfig.from_pretrained(MODEL)

    # ---- B via the class ----
    print("\n=== B: KDA class, tiled init (no LoRA) ===", flush=True)
    m = load()
    convert_to_kda(m, cfg, lora_rank=0)
    r_b = rel(ref, logits_for(m, batch))
    print(f"  relL2 {r_b:.3e}   target (monkeypatch) {TARGET_B:.3e}   "
          f"{'MATCH' if abs(r_b - TARGET_B) / TARGET_B < 0.25 else 'DIVERGES'}")
    print(f"  vs same-math control {CONTROL:.3e}: "
          f"{'PASS' if r_b <= CONTROL * 1.5 else 'FAIL'}")

    la = get_trunk(m).layers[0].linear_attn
    print(f"  layer0 type          : {type(la).__name__}")
    print(f"  in_proj_a            : {tuple(la.in_proj_a.weight.shape)}")
    print(f"  A_log / dt_bias      : {tuple(la.A_log.shape)} / {tuple(la.dt_bias.shape)}")
    del m; torch.cuda.empty_cache()

    # ---- A + B via the class ----
    print("\n=== A+B: fusion + KDA class ===", flush=True)
    m = load(); fuse_model(m, verbose=False); convert_to_kda(m, cfg, lora_rank=0, verbose=False)
    r_ab = rel(ref, logits_for(m, batch))
    print(f"  relL2 {r_ab:.3e}   target {TARGET_AB:.3e}   "
          f"{'MATCH' if abs(r_ab - TARGET_AB) / TARGET_AB < 0.25 else 'DIVERGES'}")
    del m; torch.cuda.empty_cache()

    # ---- LoRA variant must ALSO be exact at init (B = 0) ----
    print("\n=== B with zero-init LoRA (rank 32) ===", flush=True)
    m = load()
    convert_to_kda(m, cfg, lora_rank=32, verbose=False)
    r_lora = rel(ref, logits_for(m, batch))
    print(f"  relL2 {r_lora:.3e}   "
          f"{'MATCH (zero-init delta is inert)' if abs(r_lora - r_b) / r_b < 0.05 else 'DIVERGES'}")
    la = get_trunk(m).layers[0].linear_attn
    ntr = sum(p.numel() for p in la.trainable_gate_parameters())
    ndense = la.in_proj_a.weight.numel() + la.A_log.numel() + la.dt_bias.numel()
    print(f"  trainable gate params/layer: {ntr:,}  vs dense {ndense:,} "
          f"({ndense/ntr:.1f}x cheaper)")

    # ---- gradients actually flow into the new parameters ----
    print("\n=== trainability ===", flush=True)
    m.train()
    out = m(**batch).logits.float().pow(2).mean()
    out.backward()
    la = get_trunk(m).layers[0].linear_attn
    for name, p in (("a_lora_B", la.a_lora_B), ("A_log", la.A_log),
                    ("dt_bias", la.dt_bias)):
        gn = None if p.grad is None else p.grad.norm().item()
        print(f"  {name:<10} grad norm: {gn if gn is not None else 'NONE'}")
    m.eval(); m.zero_grad(set_to_none=True)
    del m; torch.cuda.empty_cache()

    # ---- RoPE-seeded init spreads the decay ----
    print("\n=== RoPE-seeded decay init ===", flush=True)
    m = load()
    la0 = get_trunk(m).layers[0].linear_attn
    alpha_before = (-la0.A_log.data.float().exp()).exp()
    print(f"  GDN alpha (scalar/head): median {alpha_before.median():.4f}  "
          f"spread {alpha_before.std():.4f}")
    convert_to_kda(m, cfg, lora_rank=32, seed_from_rope=True, verbose=False)
    la0 = get_trunk(m).layers[0].linear_attn
    alpha_after = (-la0.A_log.data.float().exp()).exp()
    print(f"  KDA alpha (per-channel): median {alpha_after.median():.4f}  "
          f"spread {alpha_after.std():.4f}  range "
          f"[{alpha_after.min():.4f}, {alpha_after.max():.4f}]")
    r_seed = rel(ref, logits_for(m, batch))
    print(f"  relL2 after seeding: {r_seed:.3e}  (expected to be NON-exact -- "
          f"seeding deliberately changes behaviour)")
    del m; torch.cuda.empty_cache()

    # ---- save / load round trip ----
    print("\n=== save / load round trip ===", flush=True)
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    m = load(); fuse_model(m, verbose=False); convert_to_kda(m, cfg, lora_rank=0, verbose=False)
    before = logits_for(m, batch)
    getattr(m.config, "text_config", m.config).kda_lift = True
    m.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    del m; torch.cuda.empty_cache()

    m2 = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.float32, device_map="cuda")
    m2.eval()
    cfg2 = AutoConfig.from_pretrained(OUT)
    lifted = getattr(getattr(cfg2, "text_config", cfg2), "kda_lift", False)
    print(f"  config records kda_lift: {lifted}")
    # the class must be re-installed on load; weights already have the lifted shapes
    convert_to_kda(m2, cfg2, lora_rank=0, verbose=False)
    after = logits_for(m2, batch)
    print(f"  round-trip relL2 vs pre-save: {rel(before, after):.3e}")
    print(f"  checkpoint size: "
          f"{sum(os.path.getsize(os.path.join(OUT,f)) for f in os.listdir(OUT))/2**30:.2f} GiB")
    del m2; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
