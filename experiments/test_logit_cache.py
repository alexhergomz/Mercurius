"""Does a cached top-k teacher reproduce the objective's terms? Measure first.

Caching the teacher is a one-shot pass over the whole corpus, so its precision
has to be settled BEFORE paying for it, not discovered afterwards. This compares
every term the trainer actually reads -- reverse KL, forward KL, and the excess
CE data term -- against the exact full-vocabulary values, over real model
log-probs rather than a synthetic distribution, because the shape of the tail is
the whole question.

    python experiments/test_logit_cache.py --k 32 64 128 --dtype u8 f16
"""
import argparse

import torch
import torch.nn.functional as F

from mercurius.paths import BASE_MODEL
from mercurius.recovery.logit_cache import (dequantize, quantize, reconstruct,
                                            topk_stats, uniform_tail_entropy)


def real_logprobs(n_tokens=512, device="cuda"):
    """Log-probs from the actual model, for two 'models': a teacher row and a
    student row. The student is the same network at a higher temperature, which
    gives a plausibly mismatched pair without loading two checkpoints."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # the unmodified base model: stage-AB is surgically altered and needs our
    # own loader, and for measuring TAIL SHAPE any real model of this family
    # answers the question
    tok = AutoTokenizer.from_pretrained(str(BASE_MODEL))
    m = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL), dtype=torch.bfloat16,
                                             device_map=device)
    m.eval()
    text = ("def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n"
            "    while lo <= hi:\n        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n            return mid\n"
            "        elif arr[mid] < target:\n            lo = mid + 1\n"
            "        else:\n            hi = mid - 1\n    return -1\n") * 12
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)
    with torch.no_grad():
        lg = m(ids).logits[0].float()
    del m
    torch.cuda.empty_cache()
    t_lp = F.log_softmax(lg, -1)
    s_lp = F.log_softmax(lg / 1.3 + 0.1 * torch.randn_like(lg), -1)
    return t_lp, s_lp, ids[0]


def terms(t_lp, s_lp, tgt):
    """The three quantities the trainer reads, in nats per position."""
    rev = (s_lp.exp() * (s_lp - t_lp)).sum(-1)          # KL(s || t)
    fwd = (t_lp.exp() * (t_lp - s_lp)).sum(-1)          # KL(t || s)
    idx = tgt.view(-1, 1)
    ce = t_lp.gather(-1, idx).squeeze(-1) - s_lp.gather(-1, idx).squeeze(-1)
    return rev, fwd, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--dtype", nargs="+", default=["u8", "f16"])
    ap.add_argument("--tokens", type=int, default=384)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading real log-probs on {dev} ...", flush=True)
    t_lp, s_lp, ids = real_logprobs(a.tokens, dev)
    V = t_lp.shape[-1]
    tgt = ids[1:].contiguous()
    t_lp, s_lp = t_lp[:-1], s_lp[:-1]
    rev0, fwd0, ce0 = terms(t_lp, s_lp, tgt)
    print(f"vocab {V:,}, {t_lp.shape[0]} positions")
    print(f"exact: revKL {rev0.mean():.4f}  fwdKL {fwd0.mean():.4f}  "
          f"excessCE {ce0.mean():.4f} nats\n")
    print(f"{'k':>5} {'dtype':>6} {'B/tok':>7} {'revKL err':>11} {'fwdKL err':>11} "
          f"{'CE err':>10} {'head mass':>10} {'tail shape':>11}")
    for k in a.k:
        idx, lp, tm, te = topk_stats(t_lp, k)
        for dt in a.dtype:
            codes, lo, sc = quantize(lp, dt)
            lp_hat = dequantize(codes, lo, sc, dt)
            t_hat = reconstruct(idx, lp_hat, tm, V, device=t_lp.device)
            rev, fwd, ce = terms(t_hat, s_lp, tgt)
            nbytes = 4 * k + (1 if dt == "u8" else 2) * k + (16 if dt == "u8" else 8)
            # how badly does a uniform tail misrepresent the real one?
            shape_gap = (uniform_tail_entropy(tm, V, k).mean() - te.mean()).abs()
            print(f"{k:>5} {dt:>6} {nbytes:>7} "
                  f"{(rev - rev0).abs().mean():>11.5f} {(fwd - fwd0).abs().mean():>11.5f} "
                  f"{(ce - ce0).abs().mean():>10.5f} {(1 - tm).mean():>10.4f} "
                  f"{shape_gap:>11.4f}")
    print("\nerrors are mean |cached - exact| in nats; compare against the exact "
          "magnitudes above.\nhead mass = probability captured by top-k; tail shape "
          "= |uniform-tail entropy - true tail entropy|.")


if __name__ == "__main__":
    main()
