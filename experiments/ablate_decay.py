"""Do we actually need to reach the trained-KDA decay distribution (alpha~0.47)?

dasc measured natively-trained KDA at median alpha 0.471, IQR [0.170, 0.774].
Our converted model sits at ~0.76 after RoPE-seeding. The open question is
whether that figure is a REQUIREMENT for long-context ability or merely a
CONSEQUENCE of how those models were trained.

This answers it without training. alpha = exp(-exp(A_log)), so a uniform shift
d on A_log maps alpha -> alpha**exp(d): one scalar moves the whole distribution.
Separately, scaling each channel's deviation from its head mean changes the
SPREAD while holding the median. Two axes, swept independently:

  axis 1  median alpha   0.90 (persistent) ... 0.30 (forgetful)
  axis 2  channel spread 0x (pure GDN) ... 3x

If perplexity and retrieval improve as alpha approaches 0.47, the target is
real and initialization should aim there. If they degrade, the target is an
artifact of native training and our model has no reason to chase it.
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
LENGTHS = [2048, 8192, 32768]


def gates(model):
    return [l.linear_attn for l in get_trunk(model).layers
            if hasattr(l, "linear_attn")]


def median_alpha(model):
    A = torch.stack([g.A_log.detach().float() for g in gates(model)])
    return (-A.exp()).exp().median().item()


def set_distribution(model, base, shift=0.0, spread=1.0):
    """A_log <- head_mean + spread*(A_log-head_mean) + shift."""
    for g, b in zip(gates(model), base):
        m = b.mean(dim=-1, keepdim=True)
        g.A_log.data.copy_((m + spread * (b - m) + shift).to(g.A_log.dtype))


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)

    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    for la in gates(model):
        la.seed_decay_from_rope()          # the init we actually use
    install_rope_dial(model, 0, "global")  # NoPE
    base = [la.A_log.detach().float().clone() for la in gates(model)]
    a0 = median_alpha(model)
    print(f"seeded init median alpha = {a0:.4f}\n", flush=True)

    results = []

    print("=== axis 1: median alpha (channel spread held at 1x) ===", flush=True)
    print(f"  {'target':<8}{'actual':<9}{'ppl@2048':>10}{'ppl@8192':>10}"
          f"{'ppl@32768':>11}{'retr@4k':>10}", flush=True)
    for target in (0.90, 0.80, a0, 0.60, 0.47, 0.30):
        # alpha_new = alpha**exp(shift)  =>  shift = ln( ln(target)/ln(a0) )
        shift = math.log(math.log(target) / math.log(a0))
        set_distribution(model, base, shift=shift, spread=1.0)
        act = median_alpha(model)
        row = {"axis": "median", "target": target, "actual": act, "spread": 1.0}
        for n in LENGTHS:
            row[f"ppl{n}"] = perplexity(model, ids, n)
            torch.cuda.empty_cache()
        _, _, row["retr"] = retrieval(model, ids, needle, 4096)
        torch.cuda.empty_cache()
        results.append(row)
        tag = "(init)" if abs(target - a0) < 1e-6 else ""
        print(f"  {target:<8.2f}{act:<9.4f}{row['ppl2048']:>10.3f}"
              f"{row['ppl8192']:>10.3f}{row['ppl32768']:>11.3f}"
              f"{row['retr']:>10.3f}  {tag}", flush=True)

    print("\n=== axis 2: channel spread (median held at init) ===", flush=True)
    print(f"  {'spread':<8}{'actual':<9}{'ppl@2048':>10}{'ppl@8192':>10}"
          f"{'ppl@32768':>11}{'retr@4k':>10}", flush=True)
    for spread in (0.0, 0.5, 1.0, 2.0, 3.0):
        set_distribution(model, base, shift=0.0, spread=spread)
        act = median_alpha(model)
        row = {"axis": "spread", "spread": spread, "actual": act}
        for n in LENGTHS:
            row[f"ppl{n}"] = perplexity(model, ids, n)
            torch.cuda.empty_cache()
        _, _, row["retr"] = retrieval(model, ids, needle, 4096)
        torch.cuda.empty_cache()
        results.append(row)
        tag = "(= pure GDN, scalar decay)" if spread == 0.0 else ""
        print(f"  {spread:<8.1f}{act:<9.4f}{row['ppl2048']:>10.3f}"
              f"{row['ppl8192']:>10.3f}{row['ppl32768']:>11.3f}"
              f"{row['retr']:>10.3f}  {tag}", flush=True)

    json.dump(results, open("logs/ablate_decay.json", "w"),
              indent=2)
    print("\nwrote logs/ablate_decay.json")


if __name__ == "__main__":
    raise SystemExit(main())
