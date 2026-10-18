"""Model-agnostic helpers: locating the decoder stack, the text config, and the normalisation conventions."""
from __future__ import annotations

import torch
import torch.nn as nn


def text_config(obj):
    """The text config of a model or config (multimodal checkpoints nest it under text_config)."""
    cfg = getattr(obj, "config", obj)
    return getattr(cfg, "text_config", None) or cfg


def trunk(model):
    """The module holding .layers / .embed_tokens / .norm / .rotary_emb."""
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and not isinstance(getattr(m, attr), (nn.ModuleList, torch.Tensor)):
            nxt = getattr(m, attr)
            if hasattr(nxt, "layers") or hasattr(nxt, "language_model") or hasattr(nxt, "model"):
                m = nxt
    if not hasattr(m, "layers"):
        raise ValueError(f"cannot find the decoder layers of {type(model).__name__}")
    return m


def rotary_dim(cfg):
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    f = float(rp.get("partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 1.0)) or 1.0)
    return int(cfg.head_dim * f)


@torch.no_grad()
def norm_kind(norm):
    """'zero' if the RMSNorm multiplies by (1 + weight) (Qwen3.5 / Qwen3-Next), 'plain' if by weight (Qwen3, Llama).
    Decided by probing the module, not by class name, so new families need no table."""
    w = norm.weight
    x = torch.randn(2, w.shape[-1], device=w.device, dtype=torch.float32)
    keep = w.detach().clone()
    try:
        w.zero_()
        y = norm(x.to(w.dtype)).float()
    finally:
        w.copy_(keep)
    return "plain" if y.abs().max() < 1e-6 else "zero"


@torch.no_grad()
def norm_gain(norm):
    """The per-channel gain the norm applies after normalisation."""
    w = norm.weight.detach().float()
    return 1.0 + w if norm_kind(norm) == "zero" else w


@torch.no_grad()
def set_norm_identity(norm):
    norm.weight.zero_() if norm_kind(norm) == "zero" else norm.weight.fill_(1.0)


class ScaleNorm(nn.Module):
    """y = g * x / rms(x) with one learned scalar g (fp32). Replaces an RMSNorm whose gain was folded into the next
    projections: exact at g = 1 (same fp32 arithmetic as the zero-centred RMSNorm at weight 0)."""

    def __init__(self, eps=1e-6, init=1.0, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.tensor(float(init), dtype=torch.float32, device=device))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * self.weight).type_as(x)

    def extra_repr(self):
        return f"scalar, eps={self.eps}"


class GatedRMSNorm(nn.Module):
    """RMSNorm over the last dim, gain w (plain), then times silu(gate). Same arithmetic as Qwen3.5's
    Qwen3_5RMSNormGated (norm in fp32, cast, gain, gate in fp32)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(dt)
        h = h * torch.nn.functional.silu(gate.float())
        return h.to(dt)


def linear_like(lin):
    """The nn.Linear holding the weight of a (possibly adapter-wrapped) projection."""
    while not isinstance(lin, nn.Linear) and hasattr(lin, "base"):
        lin = lin.base
    return lin


def effective_weight(mod):
    """Dense weight of a projection including any adapter delta, fp32."""
    if hasattr(mod, "delta_weight"):
        return linear_like(mod).weight.float() + mod.delta_weight()
    return linear_like(mod).weight.float()
