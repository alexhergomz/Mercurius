"""Prove a saved checkpoint rebuilds the model the run actually trained.

No training. The run already logged, per eval step, the exact perplexity and
top-1 of the model in memory at that moment, and the eval is bit-deterministic
on this machine (measured: 0.0000% spread over 5 repeats and 3 reloads). So the
rebuild is verifiable against a number that already exists: reconstruct the
checkpoint, re-run the trainer's OWN eval function on the trainer's OWN eval
tokens, and require a match.

This matters because the failure it catches is silent. save_state keys on
requires_grad, so anything the run froze after folding it in -- merged LoRA
bases, a fused gate adapter -- is absent from the file. Rebuilding without
replaying those steps yields a model whose every tensor loads cleanly and whose
weights are wrong: strict=False reports nothing missing, because from the
model's side nothing is. The only symptom is a perplexity that disagrees with
the run, which is exactly what this compares.

Reconstruction is possible at all because the omitted weights are a
deterministic function of files still on disk (the stage-AB checkpoint plus the
init delta), not information the run held only in memory.
"""
import argparse
import json
import sys

import torch
from transformers import AutoTokenizer
from mercurius.eval.suite import ce_and_topk
from mercurius.eval.retrieval_ab import build
from mercurius.recovery.train import CKPT, EVAL_DATA
from mercurius.paths import CACHE_DIR, CKPT_DIR

LENGTHS = [2048, 8192]
# Every correct rebuild so far has matched to EXACTLY +0.0000%, so the threshold
# should be near zero. 0.2% was loose enough to pass a rebuild that had silently
# dropped six VeRA adapters (+0.1141%), which is the whole failure this file
# exists to catch. 0.01% leaves room for bf16 reduction-order noise and nothing
# else.
TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. allvera-ce")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--init-adapters",
                    default=str(CKPT_DIR / 'adapters-combined.pt'))
    ap.add_argument("--dc", type=int, default=256)
    ap.add_argument("--covs", default=str(CACHE_DIR / 'kv_covs.pt'))
    a = ap.parse_args()

    ck = a.ckpt or f"ckpt/adapters-{a.tag}-best.pt"
    lg = a.log or f"logs/recovery-{a.tag}.json"
    run = json.load(open(lg))

    # The best checkpoint is the one with the lowest ppl at the longest length,
    # which is the same rule the trainer used to decide what to overwrite.
    key = str(max(LENGTHS))
    evals = [e for e in run["eval"] if key in e]
    want = min(evals, key=lambda e: e[key]["ppl"])
    print(f"  run's best eval: step {want['step']}, "
          f"ppl@{key} {want[key]['ppl']:.6f}", flush=True)

    m = build(ck, a.dc, a.covs, init_adapters=a.init_adapters)
    tok = AutoTokenizer.from_pretrained(CKPT)
    ids = tok(open(EVAL_DATA).read(), return_tensors="pt").input_ids[0]

    print(f"\n  {'len':>6} {'run ppl':>12} {'rebuilt':>12} {'rel':>9}   "
          f"{'run top1':>9} {'rebuilt':>9}")
    ok = True
    for n in LENGTHS:
        with torch.no_grad():
            got = ce_and_topk(m, ids, n)
        torch.cuda.empty_cache()
        ref = want[str(n)]
        rel = got["ppl"] / ref["ppl"] - 1
        flag = "" if abs(rel) <= TOL else "   <-- MISMATCH"
        if abs(rel) > TOL:
            ok = False
        print(f"  {n:>6} {ref['ppl']:>12.6f} {got['ppl']:>12.6f} "
              f"{rel:>+8.4%}   {ref['top1']:>8.3f}% {got['top1']:>8.3f}%{flag}",
              flush=True)

    print()
    if ok:
        print("  PASS: the checkpoint rebuilds the trained model. No retraining "
              "is needed to evaluate it.")
    else:
        print("  FAIL: the rebuild is not the model the run trained. Something "
              "the run folded in and froze is neither in the checkpoint nor "
              "replayed by build().")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
