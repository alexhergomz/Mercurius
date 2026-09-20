"""RULER-style NIAH evaluation of the converted model against the original.

Two metrics per sample, deliberately:

  EM   RULER's own string_match_all on greedy continuation -- the fraction of
       gold strings that appear in the generated text. Comparable in KIND to
       published RULER numbers. Its floor is 0, and a 0.8B model can sit on
       that floor, at which point it separates nothing.

  NLL  mean negative log-likelihood of the gold answer span, teacher-forced
       after the same prompt. Graded, no floor, one forward. This is what
       resolves arms that EM cannot.

Both are needed. EM alone can report 0.00 for every arm and hide a real
difference; NLL alone is not comparable to anything published.

The original is ONE ARM, not a gate. An earlier version of this plan screened
tasks by whether the original scored above floor, which silently assumes the
original bounds the converted model. It does not: the converted model is 12.7%
BETTER in perplexity (findings 0.3). A task where the original floors can still
be one where a converted arm does not, and screening on the original would
discard exactly that result. Screen on the MAX over arms, or not at all.

Every arm sees byte-identical samples: they are generated once, before any model
is built, and reused.
"""
import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from mercurius.eval.retrieval_ab import build, build_original
from mercurius.recovery.train import CKPT
import ruler_gen as R
from mercurius.paths import CACHE_DIR, CKPT_DIR

DEFAULT_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3",
                 "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                 "niah_multivalue", "niah_multiquery"]


def gold_string(outputs):
    """How the answer_prefix would naturally continue."""
    if len(outputs) == 1:
        return " " + outputs[0]
    return " " + ", ".join(outputs[:-1]) + ", and " + outputs[-1]


@torch.no_grad()
def score_sample(model, tok, s, gen_tokens):
    prompt_ids = tok(s["input"] + s["answer_prefix"]).input_ids
    gold = gold_string(s["outputs"])
    gold_ids = tok(gold, add_special_tokens=False).input_ids
    if gen_tokens < 0:
        # Budget the generation to the ANSWER, per sample. A flat cap silently
        # truncates the long answers: a uuid is ~25 tokens and four numbers with
        # separators is ~40, so a 24-token cap reported EM 0.00% on
        # niah_single_3 while the teacher-forced NLL was 0.064 -- the model had
        # the answer and the metric could not see it. RULER's flat 128 is safe
        # but pays 128 decode steps on every sample including the ones that need
        # 8. Slack of 16 covers a short preamble before the answer.
        gen_tokens = len(gold_ids) + 16
    # Concatenate TOKEN ids, not strings: tokenizing the joined text can merge
    # across the boundary and the scored span would not be the gold span.
    ids = torch.tensor([prompt_ids + gold_ids], device="cuda")
    np_ = len(prompt_ids)
    # Only the gold positions are scored, so only they need logits. Without this
    # the lm_head runs over the whole sequence: 32% of the forward's FLOPs and a
    # 3.8 GiB tensor at 8k, 7.6 GiB at 16k, of which ~40 rows are read.
    # logits_to_keep=K returns the LAST K positions, so K = len(gold)+1 gives
    # positions np_-1 .. end, and [:-1] are exactly the predictors of the gold.
    k = len(gold_ids) + 1
    out = model(input_ids=ids, logits_to_keep=k)
    sl = out.logits[0, :-1].float()
    assert sl.shape[0] == len(gold_ids), (sl.shape, len(gold_ids))
    nll = F.cross_entropy(sl, ids[0, np_:], reduction="mean").item()
    del out
    torch.cuda.empty_cache()

    pred = ""
    if gen_tokens:
        pi = torch.tensor([prompt_ids], device="cuda")
        g = model.generate(pi, max_new_tokens=gen_tokens, do_sample=False,
                           use_cache=True, pad_token_id=tok.eos_token_id)
        pred = tok.decode(g[0, pi.shape[1]:], skip_special_tokens=True)
        del g
        torch.cuda.empty_cache()
    return nll, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="tag=path entries")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--lengths", nargs="+", type=int, default=[4096, 8192])
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--em-samples", type=int, default=0, metavar="M",
                    help="score EM on only the first M samples of each cell "
                         "(0 = all). EM needs a second pass over the prompt plus "
                         "decode steps, so it is roughly half the runtime, and it "
                         "is saturated at 100%% in most cells while NLL is not.")
    ap.add_argument("--gen-tokens", type=int, default=-1,
                    help="-1 budgets per sample from the gold length "
                         "(recommended); 0 disables EM scoring; >0 is a flat cap")
    ap.add_argument("--init-adapters",
                    default=str(CKPT_DIR / 'adapters-combined.pt'))
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs.pt'))
    ap.add_argument("--out", default="logs/ruler.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(CKPT)

    print("generating samples (shared by every arm)", flush=True)
    data = {}
    for task in a.tasks:
        for n in a.lengths:
            data[(task, n)] = R.generate(task, n, a.samples, tok, verbose=True)
    print(f"  {sum(len(v) for v in data.values())} samples\n", flush=True)

    rows = []
    for spec in a.arms:
        tag, path = spec.split("=", 1)
        t0 = time.perf_counter()
        m = (build_original() if path == "ORIGINAL"
             else build(path, a.dc, a.covs, init_adapters=a.init_adapters))
        res = {}
        for task in a.tasks:
            for n in a.lengths:
                nlls, preds, refs, oom = [], [], [], 0
                lim = a.em_samples or len(data[(task, n)])
                for si, s in enumerate(data[(task, n)]):
                    # One OOM at the longest length must not destroy the whole
                    # sweep. Record the gap and carry on.
                    try:
                        nll, pred = score_sample(
                            m, tok, s, a.gen_tokens if si < lim else 0)
                    except torch.cuda.OutOfMemoryError:
                        oom += 1
                        torch.cuda.empty_cache()
                        continue
                    nlls.append(nll)
                    if si < lim:
                        preds.append(pred); refs.append(s["outputs"])
                if not nlls:
                    res[f"{task}@{n}"] = {"nll": None, "em": None, "oom": oom}
                    print(f"  {tag:<14} {task:<16}@{n:<6} "
                          f"all {oom} samples OOMed", flush=True)
                    continue
                em = (R.string_match_all(preds, refs)
                      if a.gen_tokens and preds else None)
                res[f"{task}@{n}"] = {
                    "nll": sum(nlls) / len(nlls),
                    "em": em,
                    "n": len(nlls),
                    "n_em": len(preds),
                    "oom": oom,
                    "example_pred": preds[0][:80] if preds else "",
                }
                print(f"  {tag:<14} {task:<16}@{n:<6} "
                      f"NLL {res[f'{task}@{n}']['nll']:6.3f}"
                      + (f"   EM {em:6.2f}%" if em is not None else "")
                      + (f"   ({oom} OOM)" if oom else ""),
                      flush=True)
                # Per CELL, not per arm. Checkpointing only when an arm finished
                # meant an interruption 20 cells into a 24-cell arm lost all 20.
                json.dump(rows + [{"arm": tag, "res": res, "partial": True}],
                          open(a.out, "w"), indent=1)
        rows.append({"arm": tag, "res": res})
        # Dump after EVERY arm. A sweep this long must not be all-or-nothing.
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"  ({time.perf_counter() - t0:.0f}s, partial results in "
              f"{a.out.split('/')[-1]})\n", flush=True)
        del m
        torch.cuda.empty_cache()

    keys = [f"{t}@{n}" for t in a.tasks for n in a.lengths]
    print("\nNLL of the gold span (lower is better)")
    hdr = f"{'task':<24}" + "".join(f"{r['arm'][:11]:>12}" for r in rows)
    print(hdr); print("-" * len(hdr))
    for k in keys:
        print(f"{k:<24}" + "".join(
            f"{r['res'][k]['nll']:>12.3f}" if r["res"].get(k, {}).get("nll")
            is not None else f"{'--':>12}" for r in rows))
    if a.gen_tokens:
        print("\nEM recall, RULER string_match_all (higher is better)")
        print(hdr); print("-" * len(hdr))
        for k in keys:
            print(f"{k:<24}" + "".join(
                f"{r['res'][k]['em']:>11.2f}%" if r["res"].get(k, {}).get("em")
                is not None else f"{'--':>12}" for r in rows))
        mx = {k: max((r["res"].get(k, {}).get("em") or 0.0) for r in rows)
              for k in keys}
        dead = [k for k in keys if mx[k] == 0.0]
        if dead:
            print(f"\n  EM is 0 for EVERY arm on {len(dead)}/{len(keys)} settings "
                  f"-- those separate nothing at this model size; read NLL there.")
            print(f"  {', '.join(dead)}")
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
