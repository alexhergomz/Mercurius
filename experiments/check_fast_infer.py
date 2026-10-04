"""Equivalence check for models/fast_infer.py before any eval uses it.

For each arm, the SAME RULER samples are scored twice on the SAME model: the old path
(gold span token by token, PyTorch conv update in decode) and the fast path (gold
span in ONE chunked forward continuing the cache, Triton conv update). Must agree to
bf16-rounding level in span / value NLL, and greedy predictions must match. Also
times both.

    .venv/bin/python experiments/check_fast_infer.py
"""
import time

import torch
from transformers import AutoTokenizer

from mercurius.eval import ruler_gen as R
from mercurius.eval.retrieval_ab import build, build_original_nf4
from mercurius.eval.ruler import score_sample
from mercurius.models import fast_infer
from mercurius.recovery.train import CKPT

ARMS = [("ORIG", None),
        ("L1", ("ckpt/adapters-c0-long75-step7875.pt", "cache/mla_seqcal_a860e70514_covs.pt",
                "cache/mla_seqcal_a860e70514_groups.json"))]
CASES = [("niah_multikey_1", 4096), ("niah_multiquery", 8192), ("niah_multivalue", 16384)]


def run(m, tok, samples):
    out, t0 = [], time.perf_counter()
    for s in samples:
        out.append(score_sample(m, tok, s, -1))
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    data = {c: R.generate(c[0], c[1], 3, tok) for c in CASES}
    bad = False
    for tag, spec in ARMS:
        m = (build_original_nf4() if spec is None else
             build(spec[0], 512, spec[1], groups=spec[2], dial="c0", quantize=True,
                   merge_eval=True))
        for c, samples in data.items():
            m._supports_continuation = False
            old, t_old = run(m, tok, samples)
            fast_infer.install_fast_inference(m, verbose=False)
            new, t_new = run(m, tok, samples)
            # restore the stock forward for the next case's "old" pass
            for mod in m.modules():
                if type(mod).__name__ == "Qwen3_5GatedDeltaNet" and "forward" in mod.__dict__:
                    del mod.forward
            dn = max(abs(a[0] - b[0]) for a, b in zip(old, new))
            dv = max(abs(a[1] - b[1]) for a, b in zip(old, new))
            # Greedy text past the answer is free to diverge at bf16 rounding (the
            # budget is >= 128 tokens, answers are ~10-40): compare what EM scores
            # -- every gold value present -- and how long the outputs agree.
            em = lambda p, s: R.string_match_all([p], [s["outputs"]]) >= 99.99
            same = sum(em(a[2], s) == em(b[2], s) for a, b, s in zip(old, new, samples))
            pre = min(next((i for i, (x, y) in enumerate(zip(a[2], b[2])) if x != y),
                           min(len(a[2]), len(b[2]))) for a, b in zip(old, new))
            ok = dn < 0.02 and dv < 0.02 and same == len(old)
            bad |= not ok
            print(f"[fast] {tag:4s} {c[0]}@{c[1]}: max |dNLL| {dn:.4f}  |dNLL_v| {dv:.4f}  "
                  f"EM agree {same}/{len(old)}  common prefix >= {pre} chars  time {t_old:.1f}s -> {t_new:.1f}s  "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
        del m
        torch.cuda.empty_cache()
    print("[fast] ALL OK" if not bad else "[fast] MISMATCH -- do not use --fast-infer")


if __name__ == "__main__":
    main()
