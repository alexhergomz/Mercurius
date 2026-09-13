"""Does a weight-derived channel ASSIGNMENT beat the shared RoPE ladder?

seed_decay_from_rope broadcasts ONE 128-rung ladder across all 16 heads of all 18
layers (`centered.unsqueeze(0)`), so 288 heads receive bit-identical channel
profiles. Measured on this checkpoint: that ladder is orthogonal to every weight
statistic (|corr| < 0.01 against a noise floor of 1/sqrt(128) = 0.088), while the
key-axis statistics agree strongly with EACH OTHER -- ||W_k|| vs conv-DC +0.556,
||W_k|| vs ||W_q|| -0.596 -- and the value axis sits at the floor (+0.005) as a
negative control. So there is real structured per-channel signal in the weights,
and the shipped ladder is provably blind to all of it.

WHAT IS HELD FIXED. Every arm uses the same ladder SHAPE and the same rung
MULTISET: a rank map over the 128 channels, zero-mean per head. Consequences:

  * each head's inherited GDN decay magnitude is preserved exactly (the per-head
    scalar carries ~827x of real spread and is never touched),
  * the channel spread stays at 2.72x, the operating point the strength sweep
    validated -- ablate_decay measured 2x and 3x clearly worse,
  * so any difference between arms is attributable to ASSIGNMENT alone.

THE CONTROL IS THE POINT. `random` is the same multiset permuted arbitrarily per
head. Three seeds of it give the noise floor for "per-head diversity carrying no
information". Without it a kconv-vs-rope gap cannot be read:

    kconv > random > rope    -> the weight signal is real
    kconv ~ random > rope    -> per-head diversity alone was the win
    kconv ~ random ~ rope    -> hypothesis is dead, stop here

The SIGN is a free parameter no literature settles: does a high-energy key
channel want long or short memory? Both directions are swept.

No training anywhere here -- this measures INITIALIZATION quality only, which is
the regime where the 30x init-over-training effect lives. Mirrors ablate_decay.py.
"""
import sys, os, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.eval.characterize import perplexity, retrieval
from mercurius.paths import STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
LENGTHS = [2048, 8192, 32768]
# retrieval()'s 4th arg is a GAP, not a total length: it builds
# [lead][needle][gap filler][needle], so seq ~= gap + 160. It runs under no_grad
# but takes NO logits_to_keep, so it materializes the full (seq, 248320) logit
# tensor -- 15.2 GiB at gap=32768, the same allocation that OOM-killed the
# trainer. 16384 costs 7.7 GiB and is characterize.py's own validated ceiling
# (its GAPS stop there), while still bracketing the 13,103-token gap at which
# NoPE collapses at depth 0.9. That makes it the more targeted probe anyway.
RETR_GAPS = [4096, 16384]
OUT = "logs/ablate_init.json"


@torch.no_grad()
def perplexity_multi(model, ids, n, windows=8, chunk=2048):
    """Perplexity over several DISJOINT windows, not just ids[:n].

    characterize.perplexity scores one window starting at position 0. At 32768
    that is a single sample, and it is almost certainly what produced the
    control sigma of 1.472 in the first sweep -- a spread large enough to hide
    any plausible initialization effect. Pooling CE over w disjoint windows cuts
    the ESTIMATOR's variance by ~sqrt(w), which is far cheaper than the ~35 seeds
    needed to average away the same noise one sample at a time.

    CE is pooled across windows and exponentiated once, so this is corpus
    perplexity rather than a mean of per-window perplexities.
    """
    w = max(1, min(windows, (len(ids) - 1) // n))
    tot, cnt = 0.0, 0
    for k in range(w):
        off = k * n
        x = ids[off:off + n].unsqueeze(0).cuda()
        out = model(input_ids=x)
        logits, tgt = out.logits[0], x[0, 1:]
        for i in range(0, n - 1, chunk):
            j = min(i + chunk, n - 1)
            sl = logits[i:j].float()
            tot += F.cross_entropy(sl, tgt[i:j], reduction="sum").item()
            cnt += j - i
            del sl
        del out, logits, x
        torch.cuda.empty_cache()
    return float(torch.tensor(tot / max(cnt, 1)).exp()), w


def gates(model):
    return [l.linear_attn for l in get_trunk(model).layers
            if hasattr(l, "linear_attn")]


def median_alpha(model):
    A = torch.stack([g.A_log.detach().float() for g in gates(model)])
    return (-A.exp()).exp().median().item()


def restore(model, base):
    """Back to the pure tiled point -- the bit-exact GDN reduction.

    Restores dt_bias as well as A_log. Without that, any arm writing dt_bias
    would leak into every arm measured after it and quietly turn the sweep into
    a cumulative one -- each row reporting the sum of all preceding arms.
    """
    for g, (ba, bd) in zip(gates(model), base):
        g.A_log.data.copy_(ba.to(g.A_log.dtype))
        g.dt_bias.data.copy_(bd.to(g.dt_bias.dtype))


def head_to_head(model):
    """Mean pairwise correlation of channel profiles across heads, averaged over
    layers. 1.0 means every head got the same profile (today's behaviour)."""
    out = []
    for g in gates(model):
        A = g.A_log.detach().float()
        z = A - A.mean(1, keepdim=True)
        sd = z.std(1, keepdim=True).clamp_min(1e-9)
        z = z / sd
        H, D = A.shape
        C = (z @ z.T) / D
        out.append(C[~torch.eye(H, dtype=torch.bool, device=C.device)].mean())
    return torch.stack(out).mean().item()


def guard():
    """Never contend with a live trainer for the GPU. Matching on comm==python so
    this process (comm=python too) is excluded by pid, not by pattern -- a bare
    `grep train_recovery` self-matches the launching shell, which has cost this
    project a wasted launch already."""
    me = os.getpid()
    for ln in os.popen("ps -eo pid,comm,args --no-headers").read().splitlines():
        parts = ln.split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        if int(pid) == me or "python" not in comm:
            continue
        if "train_recovery.py" in args or "ablate_init.py" in args:
            raise SystemExit(f"refusing to start: pid {pid} is using the GPU\n  {args[:90]}")


def measure(model, ids, needle, tag, rows):
    row = {"tag": tag, "median_alpha": median_alpha(model),
           "head_to_head": head_to_head(model)}
    for n in LENGTHS:
        row[f"ppl{n}"], row[f"win{n}"] = perplexity_multi(model, ids, n, windows=4)
        torch.cuda.empty_cache()
    for gp in RETR_GAPS:
        try:
            _, _, row[f"retr_gap{gp}"] = retrieval(model, ids, needle, gp)
        except RuntimeError as e:          # torch.OutOfMemoryError subclasses this
            row[f"retr_gap{gp}"] = float("nan")
            print(f"    (retr gap={gp} failed: {str(e)[:60]})", flush=True)
        torch.cuda.empty_cache()
    rows.append(row)
    print(f"  {tag:<22}{row['median_alpha']:>8.4f}{row['head_to_head']:>9.3f}"
          + "".join(f"{row[f'ppl{n}']:>11.3f}" for n in LENGTHS)
          + "".join(f"{row[f'retr_gap{gp}']:>13.3f}" for gp in RETR_GAPS), flush=True)
    return row


def main():
    guard()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    install_rope_dial(model, 0, "global")            # NoPE, as we ship it
    base = [(la.A_log.detach().float().clone(),
             la.dt_bias.detach().float().clone()) for la in gates(model)]
    print(f"tiled init (bit-exact GDN point) median alpha = {median_alpha(model):.4f}\n",
          flush=True)

    arms = [("tiled (no ladder)", lambda la: None),
            ("rope ladder (ours)", lambda la: la.seed_decay_from_rope(target_alpha=None)),
            ("kconv asc",  lambda la: la.seed_decay_from_weights(source="kconv")),
            ("kconv desc", lambda la: la.seed_decay_from_weights(source="kconv", descending=True)),
            ("wk asc",     lambda la: la.seed_decay_from_weights(source="wk"))]
    # dt_bias is the axis native KDA actually diversifies (fla draws it
    # log-uniform over [0.001, 0.1] per channel and leaves A_log a per-head
    # scalar). Ours is tiled from GDN's per-head value: within-head std is
    # 0.000000. Until now this axis was never tested.
    arms += [("dt kconv asc",  lambda la: la.seed_decay_from_weights(
                  source="kconv", target="dt_bias")),
             ("dt kconv desc", lambda la: la.seed_decay_from_weights(
                  source="kconv", descending=True, target="dt_bias"))]
    for s in (0, 1, 2):
        arms.append((f"rand A_log s{s}",
                     lambda la, s=s: la.seed_decay_from_weights(
                         source="random", generator=torch.Generator().manual_seed(s))))
    for s in (0, 1, 2):
        arms.append((f"rand dt_bias s{s}",
                     lambda la, s=s: la.seed_decay_from_weights(
                         source="random", target="dt_bias",
                         generator=torch.Generator().manual_seed(100 + s))))

    hdr = (f"  {'arm':<22}{'alpha':>8}{'h2h':>9}"
           + "".join(f"{'ppl@' + str(n):>11}" for n in LENGTHS)
           + "".join(f"{'retr@' + str(g):>13}" for g in RETR_GAPS))
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    rows = []
    for tag, fn in arms:
        restore(model, base)
        for la in gates(model):
            fn(la)
        measure(model, ids, needle, tag, rows)

    json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

    # --- verdict: each axis scored against ITS OWN random control ---
    # A_log and dt_bias are different parameters with different sensitivities, so
    # pooling their controls would compare a weight-derived arm against noise it
    # was never exposed to.
    rope = next(r for r in rows if r["tag"].startswith("rope"))
    tiled = next(r for r in rows if r["tag"].startswith("tiled"))
    print("\n=== ppl@32768: each axis vs its own random control ===")
    print(f"  tiled (no ladder)    {tiled['ppl32768']:.3f}   (bit-exact GDN point)")
    print(f"  rope ladder (ours)   {rope['ppl32768']:.3f}   (shipped)")
    for axis, is_rnd, is_wd in (
            ("A_log",   lambda t: t.startswith("rand A_log"),
                        lambda t: t.startswith(("kconv", "wk"))),
            ("dt_bias", lambda t: t.startswith("rand dt_bias"),
                        lambda t: t.startswith("dt "))):
        rnd = [r for r in rows if is_rnd(r["tag"])]
        wd = [r for r in rows if is_wd(r["tag"])]
        if not rnd or not wd:
            continue
        mu = sum(r["ppl32768"] for r in rnd) / len(rnd)
        sd = (sum((r["ppl32768"] - mu) ** 2 for r in rnd) / max(len(rnd) - 1, 1)) ** 0.5
        best = min(wd, key=lambda r: r["ppl32768"])
        sig = abs(best["ppl32768"] - mu) / max(sd, 1e-9)
        print(f"\n  [{axis}]")
        print(f"    random control       {mu:.3f} +/- {sd:.3f}  (n={len(rnd)})")
        print(f"    best weight-derived  {best['ppl32768']:.3f}  ({best['tag']})")
        print(f"    delta {best['ppl32768'] - mu:+.3f}  = {sig:.1f} control sigma"
              f"{'   <- worth following up' if sig > 1.5 else '   <- not evidence'}")
    w = rows[0].get("win32768", "?")
    print(f"\n  each ppl@32768 pooled over {w} disjoint windows (the first sweep "
          f"used 1,\n  so its sigma of 1.472 was largely ESTIMATOR noise, not init variance).")
    print("  A gap under ~1 control sigma is not evidence of a weight signal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
