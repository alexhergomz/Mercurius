"""Likelihood-ranked multiple-choice suite -- the measurement the field actually uses.

WHY THIS EXISTS (#49). Every MLA-conversion and architecture-conversion paper measures
recovery with a 6-9 task LIKELIHOOD-RANKED MC average: TransMLA, MHA2MLA, CARE,
X-EcoMLA, Palu, MOHAWK, LoLCATs, Llamba, SUPRA and DroPE all do, and most report no
generative reasoning at all. We instead used SAMPLED GENERATIVE exact-match on ONE task
at n=1319, which gives a paired sigma of ~1.4 points on a difference -- the
highest-variance instrument in the space. #46.3 then could not resolve the 2.1 points of
capacity headroom that #45.3 showed was all that existed.

MC scoring fixes that three ways at once:
  * DETERMINISTIC. No sampling, so no temperature noise. We run GSM8K at 0.7.
  * MANY MORE ITEMS. ~17k here against 1319.
  * AVERAGED OVER TASKS, which is what the papers report and compare.
And it is CHEAPER: one forward pass per candidate continuation, no generation.

SCORING follows the lm-eval-harness convention, both variants reported because papers
differ in which they quote:
    acc       argmax over sum log P(continuation | context)
    acc_norm  argmax over that sum divided by continuation length in characters
`acc_norm` is the standard for HellaSwag/ARC/OBQA (long, uneven continuations); `acc` is
standard for WinoGrande and PIQA. Both are printed so a comparison cannot be cherry-
picked after the fact.

LICENSING. These are EVALUATION-ONLY datasets and are never trained on, which is the
same standing as WikiText-2 in docs/data_policy.md (CC-BY-SA, perplexity only). Worth
recording in data/ATTRIBUTION.md before any artifact is distributed:
    hellaswag      MIT
    ai2_arc        CC-BY-SA-4.0   <- SHARE-ALIKE. Eval only, never a training input.
    piqa           AFL-3.0
    openbookqa     Apache-2.0
    winogrande     CC-BY-4.0 (AI2)
No generated text from these is kept, so nothing share-alike propagates into a model
artifact -- but the distinction matters and is flagged deliberately rather than assumed.

    .venv/bin/python experiments/bench_mc.py \
        --arms c0-ungrouped150=ckpt/adapters-c0-ungrouped150.pt \
        --dial c0 --mla-groups cache/mla_groups_ungrouped_4096.json \
        --dc 512 --covs cache/kv_covs_4b_mix.pt
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


# task -> (hf id, config, split, builder). Each builder returns
# (context, [continuations], gold_index).
def _hellaswag(r):
    ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize() if r.get("ctx_b") else r["ctx"]
    return ctx, [" " + e for e in r["endings"]], int(r["label"])


def _arc(r):
    letters = r["choices"]["label"]
    gold = letters.index(r["answerKey"]) if r["answerKey"] in letters else 0
    return f"Question: {r['question']}\nAnswer:", \
           [" " + t for t in r["choices"]["text"]], gold


def _piqa(r):
    return f"Question: {r['goal']}\nAnswer:", \
           [" " + r["sol1"], " " + r["sol2"]], int(r["label"])


def _obqa(r):
    letters = r["choices"]["label"]
    gold = letters.index(r["answerKey"]) if r["answerKey"] in letters else 0
    return r["question_stem"], [" " + t for t in r["choices"]["text"]], gold


def _winogrande(r):
    # the standard formulation: substitute each option into the blank and score the
    # SUFFIX after it, so the two candidates share no differing prefix
    idx = r["sentence"].index("_")
    ctx1 = r["sentence"][:idx] + r["option1"]
    ctx2 = r["sentence"][:idx] + r["option2"]
    suffix = r["sentence"][idx + 1:]
    return [ctx1, ctx2], [suffix, suffix], int(r["answer"]) - 1


TASKS = {
    "hellaswag":     ("Rowan/hellaswag", None, "validation", _hellaswag),
    "arc_easy":      ("allenai/ai2_arc", "ARC-Easy", "test", _arc),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test", _arc),
    # ybisk/piqa is a SCRIPT dataset and datasets>=4 refuses those
    # ("Dataset scripts are no longer supported"). baber/piqa is the same
    # content as parquet; identical fields goal/sol1/sol2/label.
    "piqa":          ("baber/piqa", None, "validation", _piqa),
    "openbookqa":    ("allenai/openbookqa", "main", "test", _obqa),
    "winogrande":    ("allenai/winogrande", "winogrande_xl", "validation", _winogrande),
}


@torch.no_grad()
def score_many(model, tok, pairs, device="cuda", batch=32, pad_id=0):
    """Summed logprob of each continuation given its context, BATCHED.

    One forward per candidate was ~40 min/arm at 50k candidates, i.e. 12 h for a
    full sweep. Batching is ~10x. RIGHT padding is safe here: with causal
    attention a position only attends to earlier ones, so trailing pads cannot
    affect the scored positions. The attention mask is passed anyway.

    Returns [(logprob_sum, continuation_char_len)] in the input order.
    """
    out = []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        encs = []
        for ctx, cont in chunk:
            c = tok(ctx, add_special_tokens=False).input_ids
            f = tok(ctx + cont, add_special_tokens=False).input_ids
            encs.append((len(c), f, max(len(cont), 1)))
        L = max(len(f) for _, f, _ in encs)
        ids = torch.full((len(encs), L), pad_id, dtype=torch.long)
        msk = torch.zeros((len(encs), L), dtype=torch.long)
        for i, (_, f, _) in enumerate(encs):
            ids[i, :len(f)] = torch.tensor(f, dtype=torch.long)
            msk[i, :len(f)] = 1
        logits = model(input_ids=ids.to(device),
                       attention_mask=msk.to(device)).logits.float()
        lp = F.log_softmax(logits[:, :-1], -1)
        for i, (n_ctx, f, nchar) in enumerate(encs):
            if len(f) <= n_ctx:                  # continuation tokenised away
                out.append((-1e9, nchar))
                continue
            tgt = torch.tensor(f[1:len(f)], dtype=torch.long, device=device)
            tl = lp[i, :len(f) - 1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            out.append((float(tl[n_ctx - 1:].sum()), nchar))
    return out


def run_task(model, tok, name, limit=0, batch=32, items_per_call=8):
    from datasets import load_dataset
    hf, cfg, split, build = TASKS[name]
    ds = load_dataset(hf, cfg, split=split) if cfg else load_dataset(hf, split=split)
    rows = list(ds)[:limit] if limit else list(ds)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    n_acc = n_norm = 0
    t0 = time.time()
    for s0 in range(0, len(rows), items_per_call):
        block = rows[s0:s0 + items_per_call]
        pairs, spans, golds = [], [], []
        for r in block:
            ctx, conts, gold = build(r)
            ctxs = ctx if isinstance(ctx, list) else [ctx] * len(conts)
            spans.append((len(pairs), len(pairs) + len(conts)))
            pairs += list(zip(ctxs, conts))
            golds.append(gold)
        sc = score_many(model, tok, pairs, batch=batch, pad_id=pad)
        for (lo, hi), gold in zip(spans, golds):
            g = sc[lo:hi]
            n_acc += int(max(range(len(g)), key=lambda j: g[j][0]) == gold)
            n_norm += int(max(range(len(g)), key=lambda j: g[j][0] / g[j][1]) == gold)
        done = min(s0 + items_per_call, len(rows))
        if done % 1000 < items_per_call:
            print(f"    {name} {done}/{len(rows)}  acc {100*n_acc/done:.1f}%  "
                  f"acc_norm {100*n_norm/done:.1f}%  {(time.time()-t0)/60:.1f} min",
                  flush=True)
    return {"n": len(rows), "acc": n_acc / len(rows), "acc_norm": n_norm / len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="tag=path, or tag=ORIGINAL")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--limit", type=int, default=0, help="items per task, 0 = all")
    ap.add_argument("--dc", type=int, default=512)
    ap.add_argument("--covs", default="cache/kv_covs_4b_mix.pt")
    ap.add_argument("--mla-groups", default=None)
    ap.add_argument("--dial", default="nope",
                    choices=["nope", "k4", "k8", "c1", "k24", "c0"],
                    help="MUST match the arm's training --dial. build() defaults to "
                         "nope, so omitting this silently evaluates a c0 arm with "
                         "NoPE installed -- the hole bench_full.py and ruler.py had.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="logs/bench_mc.json")
    a = ap.parse_args()

    from mercurius.eval.retrieval_ab import build, build_original, build_original_nf4
    from transformers import AutoTokenizer
    from mercurius.recovery.train import CKPT
    tok = AutoTokenizer.from_pretrained(CKPT)

    all_res = {}
    for arm in a.arms:
        tag, path = arm.split("=", 1)
        print(f"=== {tag} ({path})  dial={a.dial}", flush=True)
        m = (build_original() if path == "ORIGINAL"
             else build_original_nf4() if path == "ORIGINAL_NF4"
             else build(path, a.dc, a.covs, groups=a.mla_groups, dial=a.dial))
        m.eval()
        res = {}
        for t in a.tasks:
            res[t] = run_task(m, tok, t, a.limit, batch=a.batch)
            r = res[t]
            print(f"  {tag} {t}: acc {100*r['acc']:.2f}%  "
                  f"acc_norm {100*r['acc_norm']:.2f}%  (n={r['n']})", flush=True)
        tot = sum(v["n"] for v in res.values())
        avg = sum(v["acc"] for v in res.values()) / len(res)
        avgn = sum(v["acc_norm"] for v in res.values()) / len(res)
        res["_avg"] = {"acc": avg, "acc_norm": avgn, "n_total": tot}
        print(f"  {tag} AVERAGE over {len(a.tasks)} tasks: acc {100*avg:.2f}%  "
              f"acc_norm {100*avgn:.2f}%  ({tot:,} items)", flush=True)
        all_res[tag] = res
        del m
        torch.cuda.empty_cache()
        with open(a.out, "w") as fh:
            json.dump(all_res, fh, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
