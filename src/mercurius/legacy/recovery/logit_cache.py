"""Cached teacher distributions: top-k, plus what the tail needs to be honest.

The teacher forward is the dominant cost of a training step -- roughly 54 GFLOP
per token for the 27B against 24 for the student's forward and backward -- and
its output never changes and never receives gradient. So it can be computed once
and replayed. The saving repeats on every later run over the same corpus, which
is the real argument: runs B through E each paid for the same teacher passes.

WHAT IS STORED, and why each piece is needed

Top-k alone is not enough. Truncating the distribution and renormalising the
survivors silently moves the tail's mass onto the head, which biases every term
that reads the teacher. So each row keeps:

    idx       (k,)  top-k token ids
    lp        (k,)  their log-probabilities, from the FULL softmax
    tail_mass scalar   1 - sum(exp(lp)), the probability the tail really holds
    tail_ent  scalar   -sum_tail p log p, the tail's entropy contribution

With mass and entropy the tail can be modelled as a blob of known size and known
spread instead of being dropped. Reconstruction spreads `tail_mass` over the
V - k unlisted ids, which is exact in mass, and `tail_ent` says how wrong the
uniform shape is -- a diagnostic that costs 4 bytes and tells us whether k is
large enough for this corpus rather than leaving us to assume.

QUANTIZATION

`lp` is stored as 8-bit codes against a per-row affine scale, because the values
that matter are close to the top and a per-row range is tight. Whether 8 bits is
ENOUGH is a measurement, not an assumption -- experiments/test_logit_cache.py
compares the resulting KL and CE terms against the exact ones, and `dtype="f16"`
is there for when it is not.

Cost per token at k=64: 64 ids (uint32) + 64 codes (uint8) + 3 scalars (f16)
= 326 bytes. 10M tokens is 3.3 GB.
"""
import json
import os

import numpy as np
import torch

MAGIC = "mercurius-logit-cache-v1"


def topk_stats(logprobs, k=64):
    """Full-vocabulary log-probs -> (idx, lp, tail_mass, tail_ent).

    Takes LOG-PROBS, not logits, so the caller has already paid the log_softmax
    and the tail statistics are computed against the true normalisation.
    """
    lp_k, idx = torch.topk(logprobs, k, dim=-1)
    head = lp_k.exp().sum(-1)
    tail_mass = (1.0 - head).clamp_min(0.0)
    # -sum_tail p log p, computed on the full row without materialising a mask
    ent_all = -(logprobs.exp() * logprobs).sum(-1)
    ent_head = -(lp_k.exp() * lp_k).sum(-1)
    tail_ent = (ent_all - ent_head).clamp_min(0.0)
    return idx, lp_k, tail_mass, tail_ent


def quantize(lp, dtype="u8"):
    """Per-row affine quantization of the top-k log-probs."""
    if dtype == "f16":
        return lp.to(torch.float16), None, None
    hi = lp.max(-1, keepdim=True).values
    lo = lp.min(-1, keepdim=True).values
    scale = (hi - lo).clamp_min(1e-6) / 255.0
    codes = ((lp - lo) / scale).round().clamp(0, 255).to(torch.uint8)
    return codes, lo.squeeze(-1).to(torch.float32), scale.squeeze(-1).to(torch.float32)


def dequantize(codes, lo, scale, dtype="u8"):
    if dtype == "f16":
        return codes.float()
    return codes.float() * scale.unsqueeze(-1) + lo.unsqueeze(-1)


def reconstruct(idx, lp, tail_mass, vocab, device=None, floor=-30.0):
    """Approximate full-vocabulary log-probs from a cached row.

    The tail is spread UNIFORMLY over the V - k unlisted ids. That is exact in
    total mass -- which is what keeps the head unbiased -- and wrong in shape.
    `tail_ent` (stored, not used here) measures how wrong: compare it against
    the uniform tail's entropy, tail_mass * log((V-k)/tail_mass).
    """
    device = device or lp.device
    n, k = idx.shape
    per = (tail_mass / max(vocab - k, 1)).clamp_min(1e-12)
    full = torch.full((n, vocab), 0.0, device=device, dtype=torch.float32)
    full += per.unsqueeze(-1)
    full.scatter_(-1, idx.to(device), lp.exp().to(device))
    return full.clamp_min(torch.tensor(floor, device=device).exp()).log()


def uniform_tail_entropy(tail_mass, vocab, k):
    """Entropy a uniform tail of this mass would have, for comparison with the
    stored tail_ent. A large gap means k is too small for this corpus."""
    m = tail_mass.clamp_min(1e-12)
    return m * (torch.log(torch.tensor(float(vocab - k))) - torch.log(m))


class LogitCache:
    """Append rows, write one shard per file, read back memory-mapped."""

    def __init__(self, path, k=64, vocab=None, dtype="u8"):
        self.path, self.k, self.vocab, self.dtype = path, k, vocab, dtype
        self._idx, self._lp, self._lo, self._sc, self._tm, self._te = ([] for _ in range(6))
        self._n = 0

    def add(self, logprobs):
        idx, lp, tm, te = topk_stats(logprobs, self.k)
        codes, lo, sc = quantize(lp, self.dtype)
        self._idx.append(idx.to(torch.int32).cpu().numpy())
        self._lp.append(codes.cpu().numpy())
        if lo is not None:
            self._lo.append(lo.cpu().numpy())
            self._sc.append(sc.cpu().numpy())
        self._tm.append(tm.to(torch.float32).cpu().numpy())
        self._te.append(te.to(torch.float32).cpu().numpy())
        self._n += idx.shape[0]
        if self.vocab is None:
            self.vocab = logprobs.shape[-1]

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        blob = {"idx": np.concatenate(self._idx), "lp": np.concatenate(self._lp),
                "tail_mass": np.concatenate(self._tm),
                "tail_ent": np.concatenate(self._te)}
        if self._lo:
            blob["lo"] = np.concatenate(self._lo)
            blob["scale"] = np.concatenate(self._sc)
        np.savez(self.path, **blob)
        meta = {"magic": MAGIC, "k": self.k, "vocab": self.vocab,
                "dtype": self.dtype, "rows": self._n}
        json.dump(meta, open(self.path + ".meta.json", "w"), indent=1)
        return meta

    @staticmethod
    def load(path, device="cpu"):
        meta = json.load(open(path + ".meta.json"))
        assert meta["magic"] == MAGIC, f"not a logit cache: {path}"
        z = np.load(path if path.endswith(".npz") else path + ".npz")
        t = lambda a, d: torch.from_numpy(np.asarray(a)).to(device=device, dtype=d)
        out = {"idx": t(z["idx"], torch.long),
               "tail_mass": t(z["tail_mass"], torch.float32),
               "tail_ent": t(z["tail_ent"], torch.float32), **meta}
        if meta["dtype"] == "u8":
            out["lp"] = dequantize(t(z["lp"], torch.uint8), t(z["lo"], torch.float32),
                                   t(z["scale"], torch.float32))
        else:
            out["lp"] = t(z["lp"], torch.float32)
        return out

    def bytes_per_token(self):
        idx = 4 * self.k
        lp = (1 if self.dtype == "u8" else 2) * self.k
        scal = 4 * (2 + (2 if self.dtype == "u8" else 0))
        return idx + lp + scal


# ---------------------------------------------------------------------------
# RECONSTRUCTED API -- read this before relying on it.
#
# This module previously held the project's top-k logit cache: build_cache,
# load_cache, topk_kl, taid_kl, topk_kl_terms, taid_kl_terms. It was overwritten
# by accident and there is no git history in this tree to restore from, so what
# follows is a reconstruction from the call sites in recovery/train.py and from
# the semantics its docstrings describe. It is NOT the original code and should
# be treated as such until checked against a run.
#
# The cached path is DEPRECATED here, which is why losing it was recoverable at
# all. train.py's own notes record why: at K=12 the cached gradient was worse
# than not distilling; at K=64 the true next token fell outside the cache for
# ~10.8% of positions, where the loss was silent; and topk_kl renormalised BOTH
# sides over the same 64 columns, so a student placing 1% of its mass on the
# support and 99% on garbage scored zero loss whenever the shape matched. Every
# run since uses --live-teacher. Today's independent measurement (D12, and see
# test_logit_cache.py) reproduced the same conclusion from the other direction:
# no fixed-support cache can be unbiased for reverse KL.
# ---------------------------------------------------------------------------


def topk_kl_terms(s_sel, tv_c):
    """Forward KL over the cached support only, both sides renormalised there.

    s_sel: (C, k) student logits gathered at the teacher's top-k indices.
    tv_c:  (C, k) cached teacher values for those indices (log-probs).

    The renormalisation is the documented weakness, not an oversight: mass the
    student puts outside the support is invisible here.
    """
    import torch.nn.functional as F
    t_lp = F.log_softmax(tv_c.float(), -1)
    s_lp = F.log_softmax(s_sel.float(), -1)
    return (t_lp.exp() * (t_lp - s_lp)).sum(-1)


def taid_kl_terms(s_sel, tv_c, lam):
    """TAID over the cached support: the target interpolates from the student's
    own (detached) distribution toward the cached teacher as lam -> 1."""
    import torch.nn.functional as F
    t_lp = F.log_softmax(tv_c.float(), -1)
    s_lp = F.log_softmax(s_sel.float(), -1)
    if lam < 1.0:
        m = (1.0 - lam) * s_lp.detach().exp() + lam * t_lp.exp()
        t_lp = m.clamp_min(1e-9).log()
    return (t_lp.exp() * (t_lp - s_lp)).sum(-1)


def topk_kl(h_c, W_lm, tv_c, ti_c):
    s_sel = (h_c.unsqueeze(1) * W_lm[ti_c.long()].to(h_c.dtype)).sum(-1)
    return topk_kl_terms(s_sel, tv_c)


def taid_kl(h_c, W_lm, tv_c, ti_c, lam):
    s_sel = (h_c.unsqueeze(1) * W_lm[ti_c.long()].to(h_c.dtype)).sum(-1)
    return taid_kl_terms(s_sel, tv_c, lam)


def build_cache(teacher, train_ids, path, k=64, seq=8192, n_tokens=None):
    """Teacher top-k over a token stream, written as one .pt file.

    Deprecated (see the note above). Kept so `--build-cache` does not crash.
    """
    import torch
    import torch.nn.functional as F
    dev = next(teacher.parameters()).device
    idx, val, done = [], [], 0
    limit = n_tokens or len(train_ids)
    with torch.no_grad():
        for start in range(0, min(limit, len(train_ids) - 1), seq):
            ids = train_ids[start:start + seq].unsqueeze(0).to(dev)
            if ids.shape[1] < 2:
                break
            lp = F.log_softmax(teacher(ids).logits[0].float(), -1)
            v, i = torch.topk(lp, k, dim=-1)
            idx.append(i.to(torch.int32).cpu())
            val.append(v.to(torch.float16).cpu())
            done += ids.shape[1]
    blob = {"idx": torch.cat(idx), "val": torch.cat(val), "k": k, "n_tokens": done}
    torch.save(blob, path)
    return done


def load_cache(path):
    import torch
    blob = torch.load(path, map_location="cpu")
    blob.setdefault("k", blob["idx"].shape[-1])
    blob.setdefault("n_tokens", blob["idx"].shape[0])
    return blob
