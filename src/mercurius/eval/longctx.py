"""Long-context needle-in-a-haystack, beyond the 32k we have measured.

The NoPE thesis rests on the claim that KDA layers carry position adequately at
length. Our best evidence stops at 32k with a single needle at a fixed gap. This
sweeps DEPTH x LENGTH, which is what RULER-style NIAH actually tests, and
compares the two trained arms against each other.

MEMORY. Logits are the binding term: 248,320 vocab x n positions x 2 bytes is
16 GiB at 32k and 65 GiB at 131k, which does not fit. The construction avoids
ever materializing them: the needle's SECOND occurrence is placed at the very
END of the sequence, so `logits_to_keep` returns only those positions. Depth is
then set by where the FIRST occurrence sits, which is exactly the NIAH variable.

    [ filler ][ NEEDLE ][ ................ filler ................ ][ NEEDLE ]
                  ^ depth * L                                          ^ scored

Retrieval gain = NLL(needle with no prior occurrence) - NLL(needle here). High
gain means the model found the earlier copy and copied from it.
"""
import sys, json, time, argparse, inspect, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.adapters.lora import inject_lora, freeze_base
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
NEEDLE_LEN = 16


def supports_logits_to_keep(model):
    return "logits_to_keep" in inspect.signature(model.forward).parameters


def build_model(dial, adapters=None, seed_alpha=0.0):
    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    if seed_alpha is not None:
        ta = seed_alpha if seed_alpha and seed_alpha > 0 else None
        for l in get_trunk(m).layers:
            if hasattr(l, "linear_attn"):
                l.linear_attn.seed_decay_from_rope(target_alpha=ta)
    keep, pol = {"nope": (0, "global"), "c1": (16, "local"), "c0": (32, "local")}[dial]
    install_rope_dial(m, keep, pol)
    if adapters:
        inject_lora(m, LORA_RULES, verbose=False)
        freeze_base(m)
        sd = torch.load(adapters, map_location="cpu")
        missing, unexpected = m.load_state_dict(
            {k: v.cuda() for k, v in sd.items()}, strict=False)
        unexpected = [k for k in unexpected if "lora" in k or "A_log" in k]
        if unexpected:
            raise RuntimeError(f"adapter keys with no home: {unexpected[:4]}")
        print(f"    loaded {len(sd)} adapter tensors from {adapters.split('/')[-1]}",
              flush=True)
    return m.eval()


@torch.no_grad()
def needle_at_depth(model, ids, needle, total_len, depth, use_ltk):
    """Place the needle at `depth` of the context, repeat it at the very end,
    and score only the final copy."""
    nl = needle.numel()
    filler_budget = total_len - 2 * nl
    head = max(0, int(filler_budget * depth))
    tail = filler_budget - head
    seq = torch.cat([ids[:head], needle, ids[head:head + tail], needle])
    x = seq.unsqueeze(0).cuda()

    kw = {"logits_to_keep": nl + 1} if use_ltk else {}
    out = model(input_ids=x, **kw)
    lg = out.logits[0]
    if use_ltk:
        # the kept window ends at the last position; the needle occupies the
        # final nl tokens, predicted from the nl+1 preceding logits
        sl = lg[-(nl + 1):-1].float()
    else:
        sl = lg[-(nl + 1):-1].float()
    nll = F.cross_entropy(sl, x[0, -nl:], reduction="mean").item()
    del out, lg, x
    torch.cuda.empty_cache()
    return nll


@torch.no_grad()
def needle_cold(model, ids, needle, use_ltk, ctx=512):
    """NLL of the needle with NO earlier occurrence -- the baseline the gain is
    measured against."""
    nl = needle.numel()
    seq = torch.cat([ids[:ctx], needle])
    x = seq.unsqueeze(0).cuda()
    kw = {"logits_to_keep": nl + 1} if use_ltk else {}
    lg = model(input_ids=x, **kw).logits[0]
    sl = lg[-(nl + 1):-1].float()
    nll = F.cross_entropy(sl, x[0, -nl:], reduction="mean").item()
    del lg, x
    torch.cuda.empty_cache()
    return nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[8192, 32768, 65536])
    ap.add_argument("--depths", type=float, nargs="+",
                    default=[0.05, 0.25, 0.5, 0.75, 0.95])
    ap.add_argument("--out", default="logs/longctx.json")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (NEEDLE_LEN,), generator=g)
    print(f"eval corpus {len(ids):,} tokens | needle {NEEDLE_LEN} tokens", flush=True)

    ARMS = [
        ("C0 trained",   "c0",   "ckpt/adapters-c0-control.pt"),
        ("NoPE trained", "nope", str(CKPT_DIR / 'adapters-combined.pt')),
        ("NoPE untrained", "nope", None),
    ]

    results = {}
    for name, dial, adp in ARMS:
        print(f"\n=== {name} ===", flush=True)
        m = build_model(dial, adp)
        ltk = supports_logits_to_keep(m)
        if name == ARMS[0][0]:
            print(f"    logits_to_keep supported: {ltk}", flush=True)
        cold = needle_cold(m, ids, needle, ltk)
        print(f"    cold NLL (no prior copy): {cold:.3f}", flush=True)

        rows = {}
        hdr = "    " + "depth".ljust(8) + "".join(f"{L:>10}" for L in a.lengths)
        print(hdr, flush=True)
        for d in a.depths:
            line = f"    {d:<8.2f}"
            rows[d] = {}
            for L in a.lengths:
                if len(ids) < L:
                    line += f"{'--':>10}"; continue
                try:
                    t0 = time.perf_counter()
                    nll = needle_at_depth(m, ids, needle, L, d, ltk)
                    gain = cold - nll
                    rows[d][L] = {"nll": nll, "gain": gain,
                                  "sec": time.perf_counter() - t0}
                    line += f"{gain:>10.3f}"
                except RuntimeError as e:
                    rows[d][L] = {"error": str(e)[:80]}
                    line += f"{'OOM':>10}"
                    torch.cuda.empty_cache()
            print(line, flush=True)
        results[name] = {"cold": cold, "rows": rows}
        del m
        torch.cuda.empty_cache()

    json.dump(results, open(a.out, "w"), indent=2, default=str)
    print(f"\nwrote {a.out}")

    print("\n=== retrieval gain, NoPE trained minus C0 trained ===")
    print("    (negative = NoPE worse; this is the number the thesis rests on)")
    c0, np_ = results.get("C0 trained"), results.get("NoPE trained")
    if c0 and np_:
        print("    " + "depth".ljust(8) + "".join(f"{L:>10}" for L in a.lengths))
        for d in a.depths:
            line = f"    {d:<8.2f}"
            for L in a.lengths:
                x = c0["rows"].get(d, {}).get(L, {}).get("gain")
                y = np_["rows"].get(d, {}).get(L, {}).get("gain")
                line += f"{(y - x):>10.3f}" if (x is not None and y is not None) else f"{'--':>10}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
