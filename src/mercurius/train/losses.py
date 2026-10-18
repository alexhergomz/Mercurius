"""Distillation objective, computed per chunk of positions so no (T, V) logit block is ever resident.

Per position, in nats:
    divergence   D(target || student) in the chosen mode (forward KL, reverse KL, Jensen-Shannon, Jeffreys)
    data         excess cross-entropy on the true token: -log p_s(y) + log p_t(y). Zero when the student equals the
                 teacher, negative when it beats it, on the divergence's scale by construction (ce_beta = 1 is
                 principled). The teacher term carries no gradient.
    loss         divergence + ce_beta * data

TAID (Shing et al., arXiv:2501.16937) moves the target from the student's own (detached) distribution to the
teacher's as t: start -> end, adaptively. In each space one divergence degenerates to a loss-scale ramp (logit
space + reverse, prob space + forward), so the space is chosen to match: forward -> logit, reverse -> prob
(DistiLLM's skew reverse KL).

Several teachers: combine="loss" sums the per-teacher terms with the teachers' weights; combine="mixture" builds one
target, the weighted mixture of their distributions.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG2 = math.log(2.0)


def divergence(a_lp, b_lp, mode, rev_w=1.0):
    """D(target a || model b) per row, from log-probs."""
    if mode == "forward":
        return (a_lp.exp() * (a_lp - b_lp)).sum(-1)
    if mode == "reverse":
        return (b_lp.exp() * (b_lp - a_lp)).sum(-1)
    if mode == "js":
        m = torch.logaddexp(a_lp, b_lp) - LOG2
        return 0.5 * ((a_lp.exp() * (a_lp - m)).sum(-1) + (b_lp.exp() * (b_lp - m)).sum(-1))
    if mode == "jeffreys":
        return (a_lp.exp() * (a_lp - b_lp)).sum(-1) + rev_w * (b_lp.exp() * (b_lp - a_lp)).sum(-1)
    raise ValueError(mode)


def taid_target(s_lp, t_lp, lam, space):
    if lam >= 1.0:
        return t_lp
    if space == "logit":
        return F.log_softmax((1.0 - lam) * s_lp.detach() + lam * t_lp, -1)
    q = (1.0 - lam) * s_lp.detach().exp() + lam * t_lp.exp()
    return (q / q.sum(-1, keepdim=True).clamp_min(1e-9)).clamp_min(1e-9).log()


def chunk_terms(h_s, W_s, teacher_chunks, tgt, mode, lam, taid_space, combine="loss", rev_w=1.0):
    """(2, C): [divergence, excess CE] for one chunk of positions.

    h_s (C, d_s) student final states; W_s (V, d_s) student head; teacher_chunks: [(h_t (C, d_t), W_t, weight)];
    tgt (C,) true next tokens or None (no data term)."""
    V = W_s.shape[0]
    s_lp = F.log_softmax((h_s.to(W_s.dtype) @ W_s.T).float(), -1)
    t_lps = [(F.log_softmax((h.to(W.dtype) @ W[:V].T).float(), -1), w) for h, W, w in teacher_chunks]
    if combine == "mixture" and len(t_lps) > 1:
        t_lps = [(torch.logsumexp(torch.stack([lp + math.log(w) for lp, w in t_lps]), 0), 1.0)]
    div = torch.zeros(h_s.shape[0], device=h_s.device)
    data = torch.zeros_like(div)
    idx = tgt.view(-1, 1).long() if tgt is not None else None
    for t_lp, w in t_lps:
        if idx is not None:
            data = data + w * (t_lp.gather(-1, idx).squeeze(-1) - s_lp.gather(-1, idx).squeeze(-1))
        div = div + w * divergence(taid_target(s_lp, t_lp, lam, taid_space), s_lp, mode, rev_w)
    return torch.stack([div, data])


class TAIDSchedule:
    """t_next = min(end, max(t_linear(step), t + alpha * sigmoid(m) * (1 - t))), m an EMA of the relative drop of
    the distillation loss (reference: SakanaAI/TAID)."""

    def __init__(self, total, start=0.4, end=1.0, alpha=5e-4, beta=0.99):
        self.total, self.start, self.end, self.alpha, self.beta = total, start, end, alpha, beta
        self.t, self.m, self.prev = start, 0.0, None

    def update(self, step, J):
        lin = self.start + (self.end - self.start) * min(1.0, step / max(self.total, 1))
        if self.prev is not None:
            delta = (self.prev - J) / (self.prev + 1e-15)
            self.m = self.beta * self.m + (1.0 - self.beta) * delta
            self.t = min(self.end, max(lin, self.t + self.alpha / (1.0 + math.exp(-self.m)) * (1.0 - self.t)))
        else:
            self.t = min(self.end, max(lin, self.t))
        self.prev = J
        return self.t

    def state_dict(self):
        return {"t": self.t, "m": self.m, "prev": self.prev, "total": self.total}

    def load_state_dict(self, d):
        self.t, self.m, self.prev = d["t"], d["m"], d["prev"]
        self.total = d.get("total", self.total)
