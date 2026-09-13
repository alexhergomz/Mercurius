"""Does the CHANNEL SPREAD earn its place, or is a median shift enough?

The earlier ablation swept median alpha and channel spread independently and
never crossed them. That left the decisive cell untested: spread=0 (all 128
channels identical -- mathematically pure GDN scalar decay) combined with the
better median alpha=0.60.

If spread=0 @ 0.60 matches spread=1 @ 0.60 at long context, then transKDA's
channel-wise lift contributes nothing at initialization and we should simply
keep GDN decay and shift the median. If spread is required for the long-context
gain, the lift earns its place.

No training -- these are pure initialization settings.
"""
import sys, math, json, torch
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.paths import STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
LENGTHS = [2048, 32768]
SPREADS = [0.0, 0.5, 1.0]
ALPHAS = [0.76, 0.60, 0.47]


def gates(m):
    return [l.linear_attn for l in get_trunk(m).layers if hasattr(l, "linear_attn")]


def med(m):
    A = torch.stack([g.A_log.detach().float() for g in gates(m)])
    return (-A.exp()).exp().median().item()


def apply(m, base, spread, target):
    # first set spread about the head mean, then shift the median onto target
    for g, b in zip(gates(m), base):
        hm = b.mean(dim=-1, keepdim=True)
        g.A_log.data.copy_((hm + spread * (b - hm)).to(g.A_log.dtype))
    cur = med(m)
    shift = math.log(math.log(target) / math.log(cur))
    for g in gates(m):
        g.A_log.data.add_(torch.tensor(shift, dtype=g.A_log.dtype,
                                       device=g.A_log.device))


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    m = load_kda_model(CKPT, dtype=torch.bfloat16)
    # seed WITHOUT the median retarget so `base` carries only the spread shape
    for la in gates(m):
        la.seed_decay_from_rope(target_alpha=None)
    install_rope_dial(m, 0, "global")     # NoPE
    base = [la.A_log.detach().float().clone() for la in gates(m)]

    print("grid: channel spread x median alpha, no training\n")
    print(f"  {'spread':<8}{'alpha':<8}{'ppl@2048':>10}{'ppl@32768':>11}"
          f"{'retr@4k':>10}   note", flush=True)
    rows = []
    for sp in SPREADS:
        for al in ALPHAS:
            apply(m, base, sp, al)
            r = {"spread": sp, "alpha": al, "actual": med(m)}
            for n in LENGTHS:
                r[f"ppl{n}"] = perplexity(m, ids, n)
                torch.cuda.empty_cache()
            _, _, r["retr"] = retrieval(m, ids, needle, 4096)
            torch.cuda.empty_cache()
            rows.append(r)
            note = "pure GDN scalar decay" if sp == 0.0 else ""
            print(f"  {sp:<8.1f}{al:<8.2f}{r['ppl2048']:>10.3f}"
                  f"{r['ppl32768']:>11.3f}{r['retr']:>10.3f}   {note}", flush=True)

    json.dump(rows, open("logs/ablate_grid.json", "w"),
              indent=2)

    best_long = min(rows, key=lambda r: r["ppl32768"])
    gdn_best = min([r for r in rows if r["spread"] == 0.0],
                   key=lambda r: r["ppl32768"])
    print(f"\n  best overall @32768 : spread {best_long['spread']}, "
          f"alpha {best_long['alpha']} -> {best_long['ppl32768']:.3f}")
    print(f"  best GDN-like       : spread 0.0, alpha {gdn_best['alpha']} -> "
          f"{gdn_best['ppl32768']:.3f}")
    gain = (gdn_best["ppl32768"] - best_long["ppl32768"]) / gdn_best["ppl32768"] * 100
    print(f"  channel spread is worth {gain:+.2f}% ppl@32768 over scalar decay")
    print(f"  VERDICT: {'lift earns its place' if gain > 1.0 else 'median shift alone is enough -- spread adds little'}")


if __name__ == "__main__":
    raise SystemExit(main())
