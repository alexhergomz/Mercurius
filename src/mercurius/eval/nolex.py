"""NOLEX -- NO LEXical overlap: an open long-context recall benchmark.

A needle whose question shares no content word with it, so the model must infer
a latent association instead of matching a string -- extended so it can measure
a 0.8B model and diagnose a fixed-state operator.

NoLiMa (arXiv:2502.05167) showed that NIAH is easy because the needle and the
question share words, and that removing the overlap collapses long-context
performance. Its data and code are under the Adobe Research License
(non-commercial), and no permissively licensed benchmark isolates that property
-- RULER, BABILong, HELMET, InfiniteBench, LongBench and Zoology are all open but
all either match literally or vary something else. This file is written from
scratch so the result is releasable.

It is not a clone. Four changes, each aimed at a limitation that matters here:

1. PER-FACT FEASIBILITY CALIBRATION. NoLiMa needs a model that can follow an
   instruction and generate an answer, and filters its needles once, globally,
   with a 70B model. At 0.8B that combination floors: the published table's
   smallest model, Gemma 3 4B, scores 16.4 at 4K and 2.3 at 16K. Here every fact
   is first tested at MINIMAL context (needle plus question, no haystack). A fact
   the model fails there is testing its world knowledge, not its retrieval, and
   is excluded for that model. Degradation is then measured only over facts the
   model demonstrably knows -- which separates knowledge from retrieval per
   model instead of assuming a shared filter transfers.

2. FORCED CHOICE BY LIKELIHOOD, ONE FORWARD PER SAMPLE. The answer is a name
   drawn from a fixed pool of 27 that are single tokens under this tokenizer, so
   the entire candidate set is read off the final position's logits. No
   generation, no instruction-following, no judge. The candidate set is always
   10 regardless of how many needles are present, so the chance level does not
   move when the load axis does.

3. A MEASURED FLOOR, not an assumed one. Every condition is also run with the
   needles REMOVED. A model that never saw the needle does not answer at 1/10 --
   it falls back to a prior over names. That control gives the real floor, so a
   score can be read as above chance or not. NIAH and NoLiMa assume the floor.

4. A STATE-LOAD AXIS. K independent facts are planted and one is queried, for
   K = 1, 2, 4, 8. Theory for fixed-state models says recall degrades with the
   NUMBER of items held, not only with distance (Jelassi et al. arXiv:2402.01032
   on copying; Arora et al. arXiv:2312.04927 and arXiv:2402.18668 on the
   recall-state tradeoff). Softmax attention pays no such price. This model is
   18 linear-attention layers to 6 softmax layers, so this is the axis most
   likely to expose the conversion, and neither NIAH nor NoLiMa sweeps it.

The overlap ladder is graded rather than binary: L0 names the anchor verbatim
(literal match, and the feasibility control), L1 names its city (one hop), L2
names its country (two hops). The slope across levels separates "cannot find the
needle" from "cannot make the association".
"""
import argparse
import json
import random
import sys
import time

import torch
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
from mercurius.eval.nolex_data import FACTS, NAMES, NEEDLE, LEVELS, TEMPLATE
from mercurius.eval.retrieval_ab import build, build_original
from mercurius.recovery.train import CKPT
from mercurius.paths import CACHE_DIR, CKPT_DIR, DATA_DIR

HAYSTACK_SRC = str(DATA_DIR / 'fineweb_edu_long.txt')
N_CANDIDATES = 10


class Haystack:
    """Sentences shuffled so no discourse structure survives to be exploited."""

    def __init__(self, tok, n_chars=4_000_000, seed=0):
        raw = open(HAYSTACK_SRC, encoding="utf-8", errors="replace").read(n_chars)
        sents = [s.strip() for s in sent_tokenize(raw) if 40 < len(s.strip()) < 400]
        random.Random(seed).shuffle(sents)
        self.sents = sents
        self.lens = [len(tok(s + " ", add_special_tokens=False).input_ids)
                     for s in sents]

    def window(self, n_tokens, needles, depths, start):
        """Sentences totalling ~n_tokens, with each needle at its depth."""
        picked, tot, i = [], 0, start % (len(self.sents) - 4000)
        while tot < n_tokens and len(picked) < 4000:
            picked.append(self.sents[i]); tot += self.lens[i]; i += 1
        slots = sorted(((min(int(d * len(picked)), len(picked)), nd)
                        for d, nd in zip(depths, needles)),
                       key=lambda x: -x[0])
        for pos, nd in slots:
            picked.insert(pos, nd)
        return " ".join(picked)


def make_sample(rng, hay, tok, level, K, depth, length, with_needle=True):
    """K planted facts with distinct cities and countries; one is queried."""
    chosen, cities, countries = [], set(), set()
    for f in rng.sample(FACTS, len(FACTS)):
        if f[1] in cities or f[2] in countries:
            continue
        chosen.append(f); cities.add(f[1]); countries.add(f[2])
        if len(chosen) == K:
            break
    names = rng.sample(NAMES, K)
    gi = rng.randrange(K)
    anchor, city, country = chosen[gi]
    question = LEVELS[level].format(anchor=anchor, city=city, country=country)

    needles = [NEEDLE.format(name=n, anchor=f[0]) for n, f in zip(names, chosen)]
    # the queried needle sits at the requested depth; the rest are spread out
    depths = [depth if j == gi else rng.random() for j in range(K)]
    if not with_needle:
        needles, depths = [], []
    hs = hay.window(length, needles, depths, rng.randrange(10_000))
    prompt = TEMPLATE.format(haystack=hs, question=question)

    distract = [n for n in NAMES if n not in names]
    cands = names + rng.sample(distract, N_CANDIDATES - K)
    rng.shuffle(cands)
    return {"prompt": prompt, "gold": names[gi], "cands": cands,
            "level": level, "K": K, "depth": depth, "length": length,
            "fact": FACTS.index(chosen[gi])}


@torch.no_grad()
def forced_choice(model, tok, s, cand_ids):
    ids = torch.tensor([tok(s["prompt"]).input_ids], device="cuda")
    out = model(input_ids=ids)
    lg = out.logits[0, -1].float()
    del out
    torch.cuda.empty_cache()
    sel = torch.tensor([cand_ids[c] for c in s["cands"]], device="cuda")
    scores = lg[sel]
    pred = s["cands"][int(scores.argmax())]
    gi = s["cands"].index(s["gold"])
    lp = torch.log_softmax(scores, -1)
    other = torch.cat([scores[:gi], scores[gi + 1:]])
    return {"correct": float(pred == s["gold"]),
            "p_gold": float(lp[gi].exp()),
            "margin": float(scores[gi] - other.max())}


def agg(rs):
    if not rs:
        return None
    return {k: sum(r[k] for r in rs) / len(rs) for k in ("correct", "p_gold", "margin")} | {"n": len(rs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 16384])
    ap.add_argument("--loads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--depths", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--floor-samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--init-adapters",
                    default=str(CKPT_DIR / 'adapters-combined.pt'))
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs.pt'))
    ap.add_argument("--out", default="logs/nolex.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)
    cand_ids = {n: tok(" " + n, add_special_tokens=False).input_ids[0] for n in NAMES}
    assert all(len(tok(" " + n, add_special_tokens=False).input_ids) == 1
               for n in NAMES), "candidate names must be single tokens"
    print("building haystack", flush=True)
    hay = Haystack(tok)
    print(f"  {len(hay.sents):,} shuffled sentences, "
          f"{sum(hay.lens):,} tokens\n", flush=True)

    # Feasibility set: needle + question only, no haystack. One per fact.
    feas = []
    for i, f in enumerate(FACTS):
        for level in a.levels:
            q = LEVELS[level].format(anchor=f[0], city=f[1], country=f[2])
            nm = NAMES[i % len(NAMES)]
            cands = [nm] + random.Random(a.seed + i).sample(
                [x for x in NAMES if x != nm], N_CANDIDATES - 1)
            feas.append({"prompt": TEMPLATE.format(
                haystack=NEEDLE.format(name=nm, anchor=f[0]), question=q),
                "gold": nm, "cands": cands, "fact": i, "level": level})

    # The measured conditions.
    conds = []
    rng = random.Random(a.seed)
    for level in a.levels:
        for K in a.loads:
            for n in a.lengths:
                for j in range(a.samples):
                    d = a.depths[j % len(a.depths)]
                    conds.append(make_sample(rng, hay, tok, level, K, d, n))
    floors = []
    rngf = random.Random(a.seed + 7)
    for n in a.lengths:
        for j in range(a.floor_samples):
            floors.append(make_sample(rngf, hay, tok, a.levels[0], 1, 0.5, n,
                                      with_needle=False))
    print(f"{len(feas)} feasibility + {len(conds)} measured + {len(floors)} floor "
          f"= {len(feas)+len(conds)+len(floors)} forwards per arm\n", flush=True)

    rows = []
    for spec in a.arms:
        tag, path = spec.split("=", 1)
        t0 = time.perf_counter()
        m = (build_original() if path == "ORIGINAL"
             else build(path, a.dc, a.covs, init_adapters=a.init_adapters))

        ok = {}
        for s in feas:
            r = forced_choice(m, tok, s, cand_ids)
            ok.setdefault(s["level"], {})[s["fact"]] = r["correct"] == 1.0
        for level in a.levels:
            k = sum(ok[level].values())
            print(f"  {tag:<14} feasible@{level:<12} {k}/{len(FACTS)} facts",
                  flush=True)

        res, fl, skipped = {}, {}, 0
        for s in conds:
            # Gate on per-model feasibility. If this model cannot answer this
            # fact at this level with the needle RIGHT THERE and no haystack,
            # then a failure at 16k says nothing about retrieval -- it does not
            # know the association. Counting it would blend a knowledge gap into
            # a length-degradation curve, which is the confound this control
            # exists to remove.
            if not ok[s["level"]].get(s["fact"], False):
                skipped += 1
                continue
            r = forced_choice(m, tok, s, cand_ids)
            res.setdefault((s["level"], s["K"], s["length"]), []).append(r)
        if skipped:
            print(f"  {tag:<14} skipped {skipped}/{len(conds)} samples whose "
                  f"fact is not feasible for this arm", flush=True)
        for s in floors:
            fl.setdefault(s["length"], []).append(forced_choice(m, tok, s, cand_ids))

        row = {"arm": tag,
               "feasible": {lv: sum(ok[lv].values()) for lv in a.levels},
               "skipped": skipped,
               "res": {f"{lv}|{K}|{n}": agg(v) for (lv, K, n), v in res.items()},
               "floor": {str(n): agg(v) for n, v in fl.items()}}
        rows.append(row)
        json.dump(rows, open(a.out, "w"), indent=1)
        for key in sorted(row["res"]):
            v = row["res"][key]
            print(f"  {tag:<14} {key:<24} acc {v['correct']*100:6.2f}%  "
                  f"P(gold) {v['p_gold']:.3f}  margin {v['margin']:+6.2f}",
                  flush=True)
        for n in sorted(fl):
            v = row["floor"][str(n)]
            print(f"  {tag:<14} FLOOR@{n:<18} acc {v['correct']*100:6.2f}%  "
                  f"P(gold) {v['p_gold']:.3f}", flush=True)
        print(f"  ({time.perf_counter()-t0:.0f}s)\n", flush=True)
        del m
        torch.cuda.empty_cache()

    print("\naccuracy %, needle present (FLOOR = needle removed)")
    keys = sorted({k for r in rows for k in r["res"]})
    hdr = f"{'level|K|len':<24}" + "".join(f"{r['arm'][:11]:>12}" for r in rows)
    print(hdr); print("-" * len(hdr))
    for k in keys:
        print(f"{k:<24}" + "".join(
            f"{(r['res'].get(k) or {}).get('correct', 0)*100:>11.2f}%"
            for r in rows))
    print("-" * len(hdr))
    for n in sorted({int(x) for r in rows for x in r["floor"]}):
        print(f"{'FLOOR|-|'+str(n):<24}" + "".join(
            f"{(r['floor'].get(str(n)) or {}).get('correct', 0)*100:>11.2f}%"
            for r in rows))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
