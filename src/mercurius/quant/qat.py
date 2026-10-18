"""Quantization-aware training: every forward sees the deployed model.

weights   every projection in its deployed form, rounded to its format (quant/weights.py):
            adapted projections   Q(W + dW_vera)       -- the merged weight, exactly what export writes
            dense trainables      Q(W)                 -- the MLA latent maps and the RoPE key
            frozen projections    Q(W), once, in place -- e.g. the key-RMS projection
            embedding (tied head) Q(E), once, in place
          The decay / erase / write gates of GDN-2 take `qat.gates` (q8_0 by default: they feed a cumulative
          log-decay), everything else `qat.weights` (q4_0 by default).
gradient  y = x Q(W)^T + f(x.detach()) - f(x.detach()).detach(), f the trainable path: the value is exactly the
          quantized layer, the input gradient is Q(W)^T g, and the parameter gradient is the straight-through
          estimate computed in the adapter's own low-rank space -- no dense weight gradient is ever formed.
kv cache  the attention layers' `kv_quant` hook (quant/kv.py) on exactly what they cache.

Installs by rebinding forwards and setting attributes: no parameter is created or renamed, so resume files written
before QAT load unchanged, and `release_caches` frees the eval-time weight caches.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from mercurius.adapters import VeRALinear
from mercurius.arch.attention import MLAAttention, PlainAttention
from mercurius.arch.common import trunk
from .kv import make_kv_quantizer
from .weights import fake, ste

GATES = ("in_proj_a", "in_proj_be", "in_proj_bw")


def _cached(mod, key, build):
    """No-grad (eval / generation): build the quantized weight once per parameter version. With grad on, never
    hold a second copy of the weights."""
    if torch.is_grad_enabled():
        mod.__dict__.pop("_qat_cache", None)
        return build()
    c = mod.__dict__.get("_qat_cache")
    if c is None or c[0] != key:
        c = (key, build())
        mod.__dict__["_qat_cache"] = c
    return c[1]


def release_caches(model):
    n = 0
    for m in model.modules():
        n += m.__dict__.pop("_qat_cache", None) is not None
    return n


def _vera_forward(self, x):
    fmt = self._qat_fmt

    def build():
        with torch.no_grad():
            # merged exactly as export merges it (fp32 sum rounded to the base dtype), then quantized
            W = (self.base.weight.float() + self.delta_weight()).to(self.base.weight.dtype)
            return fake(W, fmt).to(x.dtype)
    W = _cached(self, (self.vera_d._version, self.vera_b._version, x.dtype), build)
    y = F.linear(x, W, None if self.base.bias is None else self.base.bias.to(x.dtype))
    if not torch.is_grad_enabled():
        return y
    h = self.adapter(x.detach())
    return y + (h - h.detach()).to(y.dtype)


def _dense_forward(self, x):
    fmt = self._qat_fmt
    if self.weight.requires_grad and torch.is_grad_enabled():
        W = ste(self.weight, fmt)
    else:
        W = _cached(self, (self.weight._version, x.dtype), lambda: fake(self.weight, fmt).to(x.dtype))
    return F.linear(x, W.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


@torch.no_grad()
def install_qat(model, q, log=print):
    """q: the `qat` config section. Idempotent."""
    counts = {"adapted": 0, "dense": 0, "frozen": 0, "gates": 0}
    tr = trunk(model)
    for layer in tr.layers:
        for name, m in layer.named_modules():
            if getattr(m, "_qat_fmt", None) is not None:
                continue
            is_gate = name.split(".")[-1] in GATES
            fmt = q["gates"] if is_gate else q["weights"]
            if isinstance(m, VeRALinear):
                m._qat_fmt = fmt
                m.forward = types.MethodType(_vera_forward, m)
                counts["gates" if is_gate else "adapted"] += 1
            elif isinstance(m, nn.Linear) and not name.endswith(".base"):     # .base: inside a VeRALinear
                if m.weight.requires_grad:
                    m._qat_fmt = fmt
                    m.forward = types.MethodType(_dense_forward, m)
                    counts["dense"] += 1
                else:
                    m.weight.copy_(fake(m.weight, fmt))
                    m._qat_fmt = fmt
                    counts["frozen"] += 1
    emb = model.get_input_embeddings()
    if not getattr(emb, "_qat_fmt", None) and q["embedding"] != "bf16":
        emb.weight.copy_(fake(emb.weight, q["embedding"]))
        emb._qat_fmt = q["embedding"]
    head = model.get_output_embeddings()
    if head is not None and head.weight.data_ptr() != emb.weight.data_ptr() and not getattr(head, "_qat_fmt", None):
        head.weight.copy_(fake(head.weight, q["lm_head"]))
        head._qat_fmt = q["lm_head"]
    kv = q["kv"]
    nkv = 0
    for layer in tr.layers:
        sa = getattr(layer, "self_attn", None)
        if isinstance(sa, MLAAttention):
            sa.kv_quant = make_kv_quantizer(kv)
            nkv += 1
        elif isinstance(sa, PlainAttention):
            sa.kv_quant = make_kv_quantizer({**kv, "head_dim": sa.D})
            nkv += 1
    log(f"  QAT: {counts['adapted']} adapted + {counts['dense']} dense + {counts['frozen']} frozen projections "
        f"@ {q['weights']}, {counts['gates']} gates @ {q['gates']}, embedding @ {q['embedding']}; "
        f"KV cache {kv['format']}" + (f" {kv['bits']}-bit" if kv["format"] != "none" else "")
        + f" on {nkv} attention layers")
    return counts
