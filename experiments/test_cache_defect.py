"""Direct test: does the seq=2048 teacher cache actively penalise retrieval?

The structural facts are already confirmed: build_cache iterates non-overlapping
2048-token blocks, and batches() samples arbitrary offsets with windows up to
32768. What that COSTS has been inferred, not measured. This measures it.

Construction, matching characterize.retrieval():

    [ lead 128 ][ NEEDLE ][ ...... gap filler ...... ][ NEEDLE ]
                                                        ^ scored

The second needle is predictable only by copying the first. We ask the SAME
teacher for its distribution over those tokens under two contexts:

  LONG  -- the entire sequence, so the first needle is visible.
  CACHE -- exactly what our cache stores for those positions:
           ids[2048*floor(p/2048) : p].  Note this is usually much LESS than
           2048 tokens: a needle landing early in a block sees only a handful.

Then the decisive question. Our loss is KL(teacher || student) over the
teacher's top-64. If the CACHE teacher is broad and wrong there, then a student
that retrieves correctly (resembling the LONG teacher) is scored WORSE than one
that mimics the uninformed CACHE teacher. If so, the objective is not merely
silent about retrieval -- it is pushing against it.

Run with no arguments. ~3 minutes, teacher only, no training.
"""
import sys, json, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from mercurius.paths import BASE_MODEL, FINEWEB, STAGE_AB

ORIG = str(BASE_MODEL)
CKPT = str(STAGE_AB)
DATA = str(FINEWEB)
BLOCK = 2048            # the cache's block size
NL = 16                 # needle length
GAPS = [256, 1024, 4096, 16384]


@torch.no_grad()
def dist_at_needle(model, seq, nl, k=64):
    """Teacher top-k distribution over the final `nl` positions of `seq`.

    logits_to_keep avoids materializing (T, 248320): at 16.5k that is 8.2 GiB.
    """
    x = seq.unsqueeze(0).cuda()
    lg = model(input_ids=x, logits_to_keep=nl + 1).logits[0][:-1].float()
    tgt = x[0, -nl:]
    nll = F.cross_entropy(lg, tgt, reduction="mean").item()
    lp = F.log_softmax(lg, -1)
    ent = float(-(lp.exp() * lp).sum(-1).mean())
    top1 = float((lg.argmax(-1) == tgt).float().mean())
    v, i = lg.topk(k, dim=-1)
    del x
    torch.cuda.empty_cache()
    return {"nll": nll, "entropy": ent, "top1": top1, "vals": v.cpu(), "idxs": i.cpu()}


def topk_kl(student_full_logits, t_vals, t_idxs):
    """Our exact training loss: both sides renormalized over the teacher's top-k."""
    s_sel = student_full_logits.gather(-1, t_idxs.cuda())
    s_lp = F.log_softmax(s_sel.float(), -1)
    t_lp = F.log_softmax(t_vals.cuda().float(), -1)
    return float((t_lp.exp() * (t_lp - s_lp)).sum(-1).mean())


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(DATA).read(), return_tensors="pt").input_ids[0]
    g = torch.Generator().manual_seed(7)
    needle = torch.randint(5000, 60000, (NL,), generator=g)

    print("loading the ORIGINAL teacher (the model that built the cache)", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        ORIG, dtype=torch.bfloat16, device_map="cuda").eval()

    print(f"\n{'gap':>7}{'ctx seen':>10}{'NLL long':>10}{'NLL cache':>11}"
          f"{'ent long':>10}{'ent cache':>11}{'top1 long':>11}{'top1 cache':>12}")
    print("-" * 82)
    rows = []
    for gap in GAPS:
        seq = torch.cat([ids[:128], needle, ids[128:128 + gap], needle])
        L = seq.numel()
        first_scored = L - NL                     # first predicted needle token
        blk = (first_scored // BLOCK) * BLOCK     # what the cache conditions on
        ctx_seen = first_scored - blk

        long = dist_at_needle(teacher, seq, NL)
        cache = dist_at_needle(teacher, seq[blk:], NL)
        rows.append({"gap": gap, "ctx_seen": ctx_seen,
                     "long": {k: v for k, v in long.items() if k != "vals" and k != "idxs"},
                     "cache": {k: v for k, v in cache.items() if k != "vals" and k != "idxs"}})
        print(f"{gap:>7}{ctx_seen:>10}{long['nll']:>10.3f}{cache['nll']:>11.3f}"
              f"{long['entropy']:>10.3f}{cache['entropy']:>11.3f}"
              f"{long['top1']:>10.1%}{cache['top1']:>12.1%}", flush=True)

        # --- which student does our loss prefer? ---
        # Build the two candidate students explicitly as full logit tensors:
        #   retrieving    = the LONG-context teacher (knows the needle)
        #   non-retrieving= the CACHE teacher        (does not)
        x_long = seq.unsqueeze(0).cuda()
        lg_long = teacher(input_ids=x_long, logits_to_keep=NL + 1).logits[0][:-1].float()
        x_cache = seq[blk:].unsqueeze(0).cuda()
        lg_cache = teacher(input_ids=x_cache, logits_to_keep=NL + 1).logits[0][:-1].float()
        loss_retrieving = topk_kl(lg_long, cache["vals"], cache["idxs"])
        loss_mimicking = topk_kl(lg_cache, cache["vals"], cache["idxs"])
        rows[-1]["loss_retrieving"] = loss_retrieving
        rows[-1]["loss_mimicking"] = loss_mimicking
        verdict = ("PENALISED" if loss_retrieving > loss_mimicking else "ok")
        print(f"{'':>7}  training loss vs the CACHE teacher:  "
              f"retrieving student {loss_retrieving:.4f}   "
              f"mimicking student {loss_mimicking:.4f}   -> retrieval {verdict}",
              flush=True)
        del x_long, x_cache, lg_long, lg_cache
        torch.cuda.empty_cache()

    json.dump(rows, open("logs/cache_defect.json", "w"),
              indent=2)
    print("\n=== summary ===")
    for r in rows:
        dn = r["cache"]["nll"] - r["long"]["nll"]
        print(f"  gap {r['gap']:>6}: teacher saw {r['ctx_seen']:>5} tokens | "
              f"NLL on the needle {r['long']['nll']:.2f} -> {r['cache']['nll']:.2f} "
              f"({dn:+.2f}) | loss penalty for retrieving "
              f"{r['loss_retrieving'] - r['loss_mimicking']:+.4f}")
    print("\n  If NLL(cache) >> NLL(long) and the retrieving student scores WORSE,")
    print("  the cached objective is actively training retrieval away.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
