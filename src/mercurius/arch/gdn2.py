"""Gated DeltaNet-2 (Zhang et al., arXiv:2605.22791): the linear-attention layer of a converted model.

    S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T,      o_t = S_t^T q_t

    g_t = -exp(A_log) * softplus(in_proj_a x + dt_bias)    channel-wise log-decay   (H x Dk)
    b_t = sigmoid(in_proj_be x)                            erase gate, key axis     (H x Dk)
    w_t = sigmoid(in_proj_bw x)                            write gate, value axis   (H x Dv)

Lifted EXACTLY from a trained Gated DeltaNet (Qwen3.5 / Qwen3-Next `linear_attn`): the scalar decay and the scalar
write strength beta are row-tiled into channel-wise gates (sigmoid / softplus are elementwise, so a repeated row
reproduces the scalar), and with `expand_qk_heads` every value head gets its own q/k head (rows duplicated, also
exact). Training then lets the channels separate.

Changes from Mercurius-1 (docs/porting_findings.md):
  * one class, named for what it is (#10: the deployed layer was called "KDA" and lifted again at load time);
  * no in-class rank-32 LoRA on the decay gate: the decay projection is adapted like every other projection (#13);
  * q/k heads optionally expanded to the value heads (#1): GDN-2's gates are per value head, so every kernel
    (fla chunk_gdn2, llama.cpp gated_delta_net) needs q/k at that count anyway; storing them expanded removes the
    per-layer repeat at run time (at the cost of the extra q/k rows);
  * the forward depends only on fla's kernels (or the PyTorch reference) and the cache's conv / recurrent state,
    not on a specific transformers release's GatedDeltaNet internals.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import GatedRMSNorm

try:                                                     # fused Triton kernels (GPU)
    from fla.ops.gdn2 import chunk_gdn2, fused_recurrent_gdn2
except Exception:                                        # pragma: no cover - CPU / no fla
    chunk_gdn2 = fused_recurrent_gdn2 = None
try:
    from fla.modules.convolution import causal_conv1d as _fla_conv
except Exception:                                        # pragma: no cover
    _fla_conv = None


def gdn2_reference(q, k, v, g, b, w, scale=None, initial_state=None, output_final_state=False,
                   use_qk_l2norm=True):
    """Token-by-token PyTorch GDN-2 (fp32). q, k, g, b: (B, T, H, Dk); v, w: (B, T, H, Dv)."""
    if use_qk_l2norm:                                    # same form as fla / transformers: x * rsqrt(|x|^2 + eps)
        q = q.float() * torch.rsqrt(q.float().pow(2).sum(-1, keepdim=True) + 1e-6)
        k = k.float() * torch.rsqrt(k.float().pow(2).sum(-1, keepdim=True) + 1e-6)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    q, k, v, g, b, w = (x.transpose(1, 2).float() for x in (q, k, v, g, b, w))
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    S = (initial_state.float().clone() if initial_state is not None
         else torch.zeros(B, H, Dk, Dv, device=v.device))
    out = torch.empty(B, H, T, Dv, device=v.device)
    for t in range(T):
        S = S * g[:, :, t].exp().unsqueeze(-1)
        kt = k[:, :, t]
        erase = torch.einsum("bhk,bhkv->bhv", b[:, :, t] * kt, S)
        S = S + kt.unsqueeze(-1) * (w[:, :, t] * v[:, :, t] - erase).unsqueeze(-2)
        out[:, :, t] = torch.einsum("bhk,bhkv->bhv", q[:, :, t] * scale, S)
    return out.transpose(1, 2), (S if output_final_state else None)


def _causal_conv(x, weight, act=True):
    """Depthwise causal conv over x (B, C, T) with kernel weight (C, K); fla's Triton kernel when available."""
    K = weight.shape[-1]
    if _fla_conv is not None and x.is_cuda:
        B, C, T = x.shape
        lim = (2 ** 31 - 1) // max(B * C, 1) - K                  # fla indexes with 32-bit offsets
        xt = x.transpose(1, 2)
        if T <= lim:
            y = _fla_conv(x=xt.contiguous(), weight=weight, bias=None, activation="silu" if act else None)[0]
        else:
            step, outs = lim - (K - 1), []
            for s in range(0, T, step):
                lo = max(0, s - (K - 1))
                yy = _fla_conv(x=xt[:, lo:s + step].contiguous(), weight=weight, bias=None,
                               activation="silu" if act else None)[0]
                outs.append(yy[:, s - lo:])
            y = torch.cat(outs, 1)
        return y.transpose(1, 2)
    y = F.conv1d(F.pad(x, (K - 1, 0)), weight.unsqueeze(1).to(x.dtype), groups=x.shape[1])
    return F.silu(y) if act else y


class GatedDeltaNet2(nn.Module):
    def __init__(self, hidden_size, num_heads, num_qk_heads, head_k_dim, head_v_dim, conv_kernel=4, eps=1e-6,
                 layer_idx=0):
        super().__init__()
        assert num_heads % num_qk_heads == 0
        self.hidden_size, self.layer_idx = hidden_size, layer_idx
        self.num_heads, self.num_qk_heads = num_heads, num_qk_heads
        self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim
        self.key_dim, self.value_dim = num_qk_heads * head_k_dim, num_heads * head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = conv_kernel
        H, Dk, Dv = num_heads, head_k_dim, head_v_dim
        self.in_proj_qkv = nn.Linear(hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, H * Dk, bias=False)
        self.in_proj_be = nn.Linear(hidden_size, H * Dk, bias=False)
        self.in_proj_bw = nn.Linear(hidden_size, H * Dv, bias=False)
        self.A_log = nn.Parameter(torch.zeros(H, Dk))
        self.dt_bias = nn.Parameter(torch.zeros(H, Dk))
        self.conv1d = nn.Parameter(torch.zeros(self.conv_dim, conv_kernel))        # depthwise kernel (C, K)
        self.norm = GatedRMSNorm(Dv, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    # ------------------------------------------------------------------------------------------------ spec
    def spec(self):
        return {"num_heads": self.num_heads, "num_qk_heads": self.num_qk_heads, "head_k_dim": self.head_k_dim,
                "head_v_dim": self.head_v_dim, "conv_kernel": self.conv_kernel_size, "eps": self.norm.eps}

    @classmethod
    def from_spec(cls, hidden_size, spec, layer_idx):
        return cls(hidden_size, spec["num_heads"], spec["num_qk_heads"], spec["head_k_dim"], spec["head_v_dim"],
                   spec["conv_kernel"], spec["eps"], layer_idx)

    # ------------------------------------------------------------------------------------------------ lift
    @classmethod
    @torch.no_grad()
    def lift(cls, gdn, layer_idx, expand_qk_heads=True):
        """Exact GDN-2 from a trained Gated DeltaNet module (Qwen3.5 / Qwen3-Next linear_attn)."""
        H, Hk = gdn.num_v_heads, gdn.num_k_heads
        Dk, Dv = gdn.head_k_dim, gdn.head_v_dim
        rep = H // Hk
        Hq = H if expand_qk_heads else Hk
        dev, dt = gdn.in_proj_qkv.weight.device, gdn.in_proj_qkv.weight.dtype
        new = cls(gdn.hidden_size, H, Hq, Dk, Dv, gdn.conv_kernel_size, gdn.norm.variance_epsilon
                  if hasattr(gdn.norm, "variance_epsilon") else getattr(gdn.norm, "eps", 1e-6), layer_idx)
        new.to(dev, dt)
        key_dim = Hk * Dk

        def expand(rows):                                 # (Hk*Dk, ...) -> (H*Dk, ...), head j <- head j // rep
            if not expand_qk_heads or rep == 1:
                return rows
            return rows.reshape(Hk, Dk, *rows.shape[1:]).repeat_interleave(rep, 0).reshape(H * Dk, *rows.shape[1:])

        Wqkv = gdn.in_proj_qkv.weight
        new.in_proj_qkv.weight.copy_(torch.cat([expand(Wqkv[:key_dim]), expand(Wqkv[key_dim:2 * key_dim]),
                                                Wqkv[2 * key_dim:]]))
        cw = gdn.conv1d.weight.squeeze(1)                 # (C, K)
        new.conv1d.copy_(torch.cat([expand(cw[:key_dim]), expand(cw[key_dim:2 * key_dim]), cw[2 * key_dim:]]))
        if getattr(gdn.conv1d, "bias", None) is not None and gdn.conv1d.bias.abs().max() > 0:
            raise NotImplementedError("conv bias")
        new.in_proj_z.weight.copy_(gdn.in_proj_z.weight)
        new.out_proj.weight.copy_(gdn.out_proj.weight)
        new.norm.weight.copy_(gdn.norm.weight)
        new.in_proj_a.weight.copy_(gdn.in_proj_a.weight.repeat_interleave(Dk, 0))
        new.A_log.copy_(gdn.A_log.float().unsqueeze(-1).expand(H, Dk))
        new.dt_bias.copy_(gdn.dt_bias.float().unsqueeze(-1).expand(H, Dk))
        new.in_proj_be.weight.copy_(gdn.in_proj_b.weight.repeat_interleave(Dk, 0))
        new.in_proj_bw.weight.copy_(gdn.in_proj_b.weight.repeat_interleave(Dv, 0))
        new.A_log.data = new.A_log.data.float()           # decay vectors stay fp32 (cumulative log-decay)
        new.dt_bias.data = new.dt_bias.data.float()
        return new

    @torch.no_grad()
    def seed_decay(self, target_alpha):
        """Shift A_log uniformly so the median per-step decay alpha = exp(g) at the typical input is target_alpha
        (alpha -> alpha ** exp(shift)). Mercurius-1 measured 0.60 as the untrained optimum on the 4B."""
        import math
        cur = float((-self.A_log.float().exp() * F.softplus(self.dt_bias.float())).exp().median())
        if 0 < cur < 1 and 0 < target_alpha < 1:
            self.A_log.add_(math.log(math.log(target_alpha) / math.log(cur)))

    # ------------------------------------------------------------------------------------------------ forward
    def _cache_conv(self, cache, x, single_step):
        """Conv with the cache's rolling input buffer (transformers' linear-attention cache API: has_previous_state,
        layers[i].conv_states, update_conv_state). Returns the activated conv output (B, C, T)."""
        i = self.layer_idx
        has = cache is not None and _has_state(cache, i)
        if has and single_step:
            st = _conv_state(cache, i)                                    # (B, C, K) raw inputs
            full = torch.cat([st[..., 1:].to(x.dtype), x], -1)
            st.copy_(full[..., -st.shape[-1]:].to(st.dtype))
            y = (full * self.conv1d.unsqueeze(0).to(x.dtype)).sum(-1, keepdim=True)
            return F.silu(y)
        pad = 0
        if has:                                                           # continuation (chunked prefill)
            prev = _conv_state(cache, i)[..., 1:].to(x.dtype).clone()
            _update_conv(cache, x, i, self.conv_kernel_size)
            x = torch.cat([prev, x], -1)
            pad = prev.shape[-1]
        elif cache is not None:
            _update_conv(cache, x, i, self.conv_kernel_size)
        y = _causal_conv(x, self.conv1d.to(x.dtype))
        return y[..., pad:] if pad else y

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 2:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
        B, T, _ = hidden_states.shape
        H, Hq, Dk, Dv = self.num_heads, self.num_qk_heads, self.head_k_dim, self.head_v_dim
        # read BEFORE the conv update: transformers marks the state as present while caching the prefill's inputs
        prior = cache_params is not None and _has_state(cache_params, self.layer_idx)
        single = prior and T == 1
        mixed = self._cache_conv(cache_params, self.in_proj_qkv(hidden_states).transpose(1, 2), single)
        mixed = mixed.transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], -1)
        q, k = q.reshape(B, T, Hq, Dk), k.reshape(B, T, Hq, Dk)
        v = v.reshape(B, T, H, Dv)
        if Hq != H:
            q, k = q.repeat_interleave(H // Hq, 2), k.repeat_interleave(H // Hq, 2)
        a = self.in_proj_a(hidden_states).view(B, T, H, Dk)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        b = self.in_proj_be(hidden_states).view(B, T, H, Dk).sigmoid()
        w = self.in_proj_bw(hidden_states).view(B, T, H, Dv).sigmoid()
        state = _rec_state(cache_params, self.layer_idx) if prior else None
        want = cache_params is not None
        if hidden_states.is_cuda and chunk_gdn2 is not None:
            kern = fused_recurrent_gdn2 if single else chunk_gdn2
            o, final = kern(q=q, k=k, v=v, g=g.contiguous(), b=b.contiguous(), w=w.contiguous(),
                            initial_state=None if state is None else state.float(), output_final_state=want,
                            use_qk_l2norm_in_kernel=True)
        else:
            o, final = gdn2_reference(q, k, v, g, b, w, initial_state=state, output_final_state=want)
        if want:
            _update_rec(cache_params, final, self.layer_idx)
        z = self.in_proj_z(hidden_states).reshape(B, T, H, Dv)
        o = self.norm(o.to(hidden_states.dtype).reshape(-1, Dv), z.reshape(-1, Dv)).reshape(B, T, -1)
        return self.out_proj(o)


# ---------------------------------------------------------------------------------------------- cache API
# transformers' hybrid cache: one place to adapt if its linear-attention state API changes.
def _has_state(cache, i):
    try:
        return bool(cache.has_previous_state(i, state_idx=0))
    except TypeError:
        return bool(cache.has_previous_state(i))


def _layer(cache, i):
    return cache.layers[i]


def _first(x):
    if isinstance(x, dict):                              # transformers 5.x: {state_idx: tensor}
        return x.get(0)
    return x[0] if isinstance(x, (list, tuple)) else x


def _conv_state(cache, i):
    return _first(_layer(cache, i).conv_states)


def _rec_state(cache, i):
    return _first(_layer(cache, i).recurrent_states)


def _update_conv(cache, x, i, K):
    try:
        cache.update_conv_state(x, i, conv_kernel_size=K)
    except TypeError:
        cache.update_conv_state(F.pad(x, (max(K - x.shape[-1], 0), 0))[..., -K:], i)


def _update_rec(cache, s, i):
    cache.update_recurrent_state(s, i)
