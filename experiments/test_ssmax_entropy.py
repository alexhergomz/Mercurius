"""Does attention actually fade with length on THIS model? Test before training.

SSMax's premise is "attention fading": as sequence length n grows, softmax over
more keys drives its maximum down and its entropy up, so heads that should sit
sharply on one retrieved token spread out instead. The fix multiplies logits by
s*log(n) to re-sharpen.

That premise is directly measurable WITHOUT any training, which is the point of
running this first. Two outcomes, both useful:

  * entropy climbs with n, and SSMax flattens it  -> the mechanism applies here,
    and a training arm is justified.
  * entropy is already flat in n                  -> the premise does not hold on
    this model and we skip the arm entirely.

Cheaper and more diagnostic than a paired training run, and it fails loudly
rather than producing an ambiguous 0.3-sigma result.

MEMORY. Uses eager attention weights, which are O(n^2): at 4096 one layer is
8 heads x 4096^2 x 4 B = 0.5 GiB at the fp32 softmax. One layer at a time, and
do not raise --max-len past 4096 without redoing that arithmetic.
"""
import sys, json, math, argparse, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora
from mercurius.adapters.ssmax import install_ssmax
from mercurius.paths import CKPT_DIR, STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
LORA_RULES = [
    ("self_attn.q_proj", 32), ("self_attn.k_proj", 32),
    ("self_attn.v_proj", 32), ("self_attn.o_proj", 32),
    ("linear_attn.out_proj", 16), ("linear_attn.in_proj_qkv", 16),
    ("mlp.gate_proj", 16), ("mlp.up_proj", 16), ("mlp.down_proj", 16),
    ("lm_head", 0), ("embed_tokens", 0),
]


def build(adapters=None):
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    for l in get_trunk(m).layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")
    if adapters:
        inject_lora(m, LORA_RULES, verbose=False)
        sd = torch.load(adapters, map_location="cpu")
        m.load_state_dict({k: v.cuda() for k, v in sd.items()}, strict=False)
    return m.eval()


@torch.no_grad()
def entropies(model, ids, n):
    """Mean causal-attention entropy per full-attention layer at length n."""
    trunk = get_trunk(model)
    x = ids[:n].unsqueeze(0).cuda()
    caps, hooks = {}, []
    for i, l in enumerate(trunk.layers):
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        c = {}
        def pre(mod, args, kwargs, _c=c):
            _c["h"] = args[0] if args else kwargs.get("hidden_states")
            return None
        hooks.append(sa.register_forward_pre_hook(pre, with_kwargs=True))
        caps[i] = c
    model(input_ids=x, logits_to_keep=1)
    for h in hooks:
        h.remove()

    out = {}
    mask = torch.full((n, n), float("-inf"), device="cuda").triu(1)
    for i, c in caps.items():
        sa = trunk.layers[i].self_attn
        h = c["h"]
        B, T, _ = h.shape
        hs = (B, T, -1, sa.head_dim)
        q, _ = torch.chunk(sa.q_proj(h).view(B, T, -1, sa.head_dim * 2), 2, dim=-1)
        q = sa.q_norm(q.view(hs)).transpose(1, 2)          # goes through SSMax if installed
        k = sa.k_norm(sa.k_proj(h).view(hs)).transpose(1, 2)
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        lg = (q.float() @ k.float().transpose(-1, -2)) * sa.scaling + mask
        p = F.softmax(lg, dim=-1)
        out[i] = float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
        del q, k, lg, p
        torch.cuda.empty_cache()
    del x, caps
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters",
                    default=str(CKPT_DIR / 'adapters-combined.pt'))
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--n-ref", type=int, default=8192)
    ap.add_argument("--out",
                    default="logs/ssmax_entropy.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    m = build(a.adapters)

    res = {}
    for tag in ("plain", "ssmax"):
        if tag == "ssmax":
            restore = install_ssmax(m, n_ref=a.n_ref, verbose=True)
        print(f"\n=== {tag} ===", flush=True)
        print("  " + "layer".ljust(8) + "".join(f"{n:>10}" for n in a.lengths), flush=True)
        per = {n: entropies(m, ids, n) for n in a.lengths}
        layers = sorted(per[a.lengths[0]])
        for li in layers:
            print(f"  {li:<8}" + "".join(f"{per[n][li]:>10.4f}" for n in a.lengths),
                  flush=True)
        mean = {n: sum(per[n].values()) / len(per[n]) for n in a.lengths}
        print("  " + "MEAN".ljust(8) + "".join(f"{mean[n]:>10.4f}" for n in a.lengths),
              flush=True)
        res[tag] = {"per_layer": {str(k): v for k, v in per.items()}, "mean": mean}
        if tag == "ssmax":
            restore()

    json.dump(res, open(a.out, "w"), indent=2, default=str)
    lo, hi = a.lengths[0], a.lengths[-1]
    dp = res["plain"]["mean"][hi] - res["plain"]["mean"][lo]
    ds = res["ssmax"]["mean"][hi] - res["ssmax"]["mean"][lo]
    print(f"\n=== does attention fade with length on this model? ===")
    print(f"  plain: entropy {res['plain']['mean'][lo]:.4f} -> "
          f"{res['plain']['mean'][hi]:.4f} over {lo}->{hi}  ({dp:+.4f})")
    print(f"  ssmax: entropy {res['ssmax']['mean'][lo]:.4f} -> "
          f"{res['ssmax']['mean'][hi]:.4f} over {lo}->{hi}  ({ds:+.4f})")
    if dp <= 0.05:
        print("  -> entropy is FLAT in length. SSMax's premise does not hold here;")
        print("     skip the training arm, the mechanism has nothing to fix.")
    elif ds < dp:
        print(f"  -> entropy climbs and SSMax reduces the climb by "
              f"{100*(1-ds/dp):.0f}%. Mechanism applies; a training arm is justified.")
    else:
        print("  -> entropy climbs but SSMax does NOT flatten it. Either the")
        print("     reference length is wrong or the parameterization is off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
