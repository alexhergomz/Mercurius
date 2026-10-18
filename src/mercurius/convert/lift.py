"""Exact (function-preserving) conversion stages.

fold_norms      every foldable RMSNorm gain moves into the projections that consume it: W' = W diag(gain), and the
                norm becomes parameter-free. Foldable = a decoder layer's input_layernorm (consumed by the token
                mixer's input projections) and post_attention_layernorm (consumed by a dense MLP's gate/up).
                Not folded: model.norm (the lm_head may be tied to the embedding), q_norm / k_norm (applied after
                the projection, before RoPE), the gated norm inside the linear-attention layer, and the
                post-attention norm of a mixture-of-experts MLP (left as is: its router and experts would all
                need the fold).
scalenorm       a folded (parameter-free) norm becomes ScaleNorm, one learned scalar: exact at 1.0.
lift_gdn2       Gated DeltaNet -> GDN-2 (arch/gdn2.py), exact, optionally with q/k expanded to the value heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from mercurius.arch.common import ScaleNorm, norm_gain, set_norm_identity, trunk
from mercurius.arch.gdn2 import GatedDeltaNet2


def _mixer_consumers(layer):
    la = getattr(layer, "linear_attn", None)
    if la is not None:
        if isinstance(la, GatedDeltaNet2):
            return [la.in_proj_qkv, la.in_proj_z, la.in_proj_a, la.in_proj_be, la.in_proj_bw]
        return [getattr(la, n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a") if hasattr(la, n)]
    sa = layer.self_attn
    return [getattr(sa, n) for n in ("q_proj", "k_proj", "v_proj") if isinstance(getattr(sa, n, None), nn.Linear)]


def _mlp_consumers(layer):
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and not hasattr(mlp, "experts"):
        return [mlp.gate_proj, mlp.up_proj]
    return None                                                 # mixture of experts: not folded


@torch.no_grad()
def _fold(norm, consumers):
    g = norm_gain(norm)
    for lin in consumers:
        assert lin.weight.shape[1] == g.shape[0]
        lin.weight.copy_((lin.weight.float() * g.to(lin.weight.device).unsqueeze(0)).to(lin.weight.dtype))
    set_norm_identity(norm)


@torch.no_grad()
def fold_norms(model, log=print):
    n, skipped = 0, 0
    for layer in trunk(model).layers:
        _fold(layer.input_layernorm, _mixer_consumers(layer))
        n += 1
        mc = _mlp_consumers(layer)
        if mc is None:
            skipped += 1
            continue
        _fold(layer.post_attention_layernorm, mc)
        n += 1
    log(f"  folded {n} norms" + (f"; {skipped} MoE post-attention norms left per-channel" if skipped else ""))
    return n


@torch.no_grad()
def to_scalenorm(model, log=print):
    """Replace every folded (identity) decoder norm with a learned scalar."""
    n = 0
    for layer in trunk(model).layers:
        for name in ("input_layernorm", "post_attention_layernorm"):
            norm = getattr(layer, name)
            if isinstance(norm, ScaleNorm):
                continue
            if (norm_gain(norm) - 1).abs().max() > 0:
                continue                                         # not folded: keep its gains
            eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))
            setattr(layer, name, ScaleNorm(eps, 1.0, norm.weight.device))
            n += 1
    log(f"  ScaleNorm on {n} folded norms")
    return n


@torch.no_grad()
def lift_gdn2(model, expand_qk_heads=True, seed_alpha=None, log=print):
    n = 0
    for i, layer in enumerate(trunk(model).layers):
        la = getattr(layer, "linear_attn", None)
        if la is None or isinstance(la, GatedDeltaNet2):
            continue
        layer.linear_attn = GatedDeltaNet2.lift(la, i, expand_qk_heads)
        if seed_alpha:
            layer.linear_attn.seed_decay(seed_alpha)
        n += 1
    if n:
        la = next(l.linear_attn for l in trunk(model).layers if hasattr(l, "linear_attn"))
        log(f"  GDN-2 on {n} linear-attention layers: {la.num_heads} heads x (Dk {la.head_k_dim}, Dv "
            f"{la.head_v_dim}), q/k heads {la.num_qk_heads}" + (f", decay retargeted to median alpha {seed_alpha}"
                                                                if seed_alpha else ""))
    return n
