"""Full characterization of Stage C options at real context lengths.

Two metrics, because they answer different questions:

  PERPLEXITY on natural text -- does the model still model language well?
  RETRIEVAL  (repeated-span)  -- can it still find something far back?

Retrieval construction: [filler][NEEDLE][filler][NEEDLE]. The NLL of the SECOND
needle occurrence drops sharply if the model can copy from the first. Reporting
NLL(first) - NLL(second) isolates retrieval from the needle's intrinsic
improbability, and sweeping the gap gives a length-resolved curve. This works on
a base model with no instruction following.

Run in bf16: we are comparing configurations, not verifying exactness, and bf16
is ~2x faster. Exactness gates use fp32 elsewhere.
"""
import sys, json, time, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.models.kda import load_kda_model
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.rope_dial import install_rope_dial
from mercurius.paths import STAGE_AB, WIKITEXT

CKPT = str(STAGE_AB)
DATA = str(WIKITEXT)
OUT = "logs/characterization.json"

# (label, keep_freqs, policy).  32 freqs == 64 rotary dims == untouched.
DIALS = [
    ("C0  full RoPE",      32, "local"),
    ("C1  32d local",      16, "local"),
    ("C1  32d global",     16, "global"),
    ("C1  32d stride",     16, "stride"),
    ("C2  16d local",       8, "local"),
    ("C2   8d local",       4, "local"),
    ("C3   0d NoPE",        0, "global"),
]
LENGTHS = [2048, 8192, 32768]
GAPS = [256, 1024, 4096, 16384]


@torch.no_grad()
def perplexity(model, ids, n, chunk=2048):
    """Chunked CE: vocab is 248,320, so .float() on full logits would allocate
    32.5 GiB at 32k positions -- on top of the 17 GiB the forward already holds."""
    x = ids[:n].unsqueeze(0).cuda()
    out = model(input_ids=x)
    logits, tgt = out.logits[0], x[0, 1:]
    tot, cnt = 0.0, 0
    for i in range(0, n - 1, chunk):
        j = min(i + chunk, n - 1)
        sl = logits[i:j].float()
        nll = F.cross_entropy(sl, tgt[i:j], reduction="sum")
        tot += nll.item(); cnt += j - i
        del sl
    del out, logits
    torch.cuda.empty_cache()
    return float(torch.tensor(tot / cnt).exp())


@torch.no_grad()
def retrieval(model, ids, needle, gap, filler_lead=128):
    """NLL(first needle) - NLL(second needle). Higher = better retrieval."""
    nl = needle.numel()
    seq = torch.cat([ids[:filler_lead], needle,
                     ids[filler_lead:filler_lead + gap], needle]).unsqueeze(0).cuda()
    out = model(input_ids=seq)
    p1, p2 = filler_lead, filler_lead + nl + gap
    # slice ONLY the needle positions before upcasting -- full-sequence .float()
    # would allocate tens of GiB against this vocabulary
    def span_nll(p):
        sl = out.logits[0, p - 1:p - 1 + nl].float()
        return F.cross_entropy(sl, seq[0, p:p + nl], reduction="mean").item()
    first, second = span_nll(p1), span_nll(p2)
    del out
    torch.cuda.empty_cache()
    return first, second, first - second


def run_config(model, ids, needle, label):
    row = {"config": label, "ppl": {}, "retrieval": {}}
    for n in LENGTHS:
        t0 = time.perf_counter()
        row["ppl"][n] = perplexity(model, ids, n)
        torch.cuda.empty_cache()
        print(f"    ppl@{n:<6} {row['ppl'][n]:8.3f}   ({time.perf_counter()-t0:5.1f}s)",
              flush=True)
    for g in GAPS:
        f, s, gain = retrieval(model, ids, needle, g)
        row["retrieval"][g] = {"first": f, "second": s, "gain": gain}
        torch.cuda.empty_cache()
        print(f"    gap {g:<6} first {f:6.3f}  second {s:6.3f}  "
              f"gain {gain:6.3f}", flush=True)
    return row


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    print(f"corpus: {len(ids):,} tokens", flush=True)

    # needle: mid-frequency token ids, unlikely as a natural sequence
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (16,), generator=g)
    print(f"needle: {needle.tolist()}\n", flush=True)

    model = load_kda_model(CKPT, dtype=torch.bfloat16)
    results = {"dial": [], "init": []}

    print("=== dial sweep (tiled KDA init) ===", flush=True)
    for label, keep, policy in DIALS:
        print(f"\n  {label}", flush=True)
        restore = install_rope_dial(model, keep, policy)
        results["dial"].append(run_config(model, ids, needle, label))
        restore()

    print("\n\n=== init A/B: tiled vs RoPE-seeded decay ===", flush=True)
    # snapshot A_log so seeding can be undone
    saved = [l.linear_attn.A_log.data.clone()
             for l in get_trunk(model).layers if hasattr(l, "linear_attn")]
    for init_name in ("tiled", "rope-seeded"):
        if init_name == "rope-seeded":
            for l in get_trunk(model).layers:
                if hasattr(l, "linear_attn"):
                    l.linear_attn.seed_decay_from_rope()
        for label, keep, policy in (("C0  full RoPE", 32, "local"),
                                    ("C3   0d NoPE", 0, "global")):
            print(f"\n  [{init_name}] {label}", flush=True)
            restore = install_rope_dial(model, keep, policy)
            r = run_config(model, ids, needle, f"{init_name} | {label}")
            r["init"] = init_name
            results["init"].append(r)
            restore()
        if init_name == "rope-seeded":
            break
        # restore tiled A_log before the seeded pass (seeding is additive)
        for l, a in zip([l for l in get_trunk(model).layers
                         if hasattr(l, "linear_attn")], saved):
            l.linear_attn.A_log.data.copy_(a)

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")

    # ---- tables ----
    print("\n\n=== PERPLEXITY ===")
    hdr = "  {:<18}" + "".join(f"{n:>10}" for n in LENGTHS)
    print(hdr.format("config", *LENGTHS) if False else
          "  {:<18}".format("config") + "".join(f"{n:>10}" for n in LENGTHS))
    for r in results["dial"]:
        print("  {:<18}".format(r["config"]) +
              "".join(f"{r['ppl'][n]:>10.3f}" for n in LENGTHS))

    print("\n=== RETRIEVAL GAIN (NLL first - second; higher is better) ===")
    print("  {:<18}".format("config") + "".join(f"{g:>10}" for g in GAPS))
    for r in results["dial"]:
        print("  {:<18}".format(r["config"]) +
              "".join(f"{r['retrieval'][g]['gain']:>10.3f}" for g in GAPS))

    print("\n=== INIT A/B ===")
    print("  {:<28}".format("config") + "".join(f"{n:>10}" for n in LENGTHS))
    for r in results["init"]:
        print("  {:<28}".format(r["config"]) +
              "".join(f"{r['ppl'][n]:>10.3f}" for n in LENGTHS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
