"""Attention layers of a converted model.

MLAAttention -- absorbable multi-head latent attention, converted from a trained GQA layer (convert/mla.py):

    cache per token:  k_rope (n_rope * rd, exact RoPE key, shared by all heads)
                      c      (r, the CARE latent: every head's NoPE keys, the RoPE-free key components, values)
                      rms    (G scalars: the original key's per-group RMS, k_norm's normaliser, kept exact)
    keys   K_g = [RoPE(k_rope) | up_k c] / rms_g     (G groups; up_k folds k_norm's gain and the RoRoPE mixing)
    values V_g = up_v c
    query  per head, after q_norm (and the per-head map R_h, kept in the RoPE commutant), the rotary dims are
           mixed per frequency by the RoRoPE axes U_l[g(h), :] (buffer `rorope`), so the score is exactly
           q_h . k_g / rms_g of the original layer when no rank is dropped and n_rope = G.

Everything the forward needs is a parameter or a persistent buffer, so a checkpoint loads without replaying the
conversion (docs/porting_findings.md #9). The KV-cache quantizer for QAT is attached as `kv_quant` and applied to
exactly what the deployed cache holds: the latent, the RoPE key AFTER RoPE (quantizing post-RoPE removes the
per-read un-rotation the pre-RoPE layout needed, #5) and the rms in fp16 (#7).

PlainAttention -- the original GQA layer (gated output when the family has one), with the same `kv_quant` hook on
post-RoPE keys and values; used when MLA conversion is off.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with either convention: y = x / rms(x) * (1 + w) (zero_centered) or * w."""

    def __init__(self, dim, eps=1e-6, zero_centered=True):
        super().__init__()
        self.zero_centered, self.eps = zero_centered, eps
        self.weight = nn.Parameter(torch.zeros(dim) if zero_centered else torch.ones(dim))

    def forward(self, x):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.zero_centered:
            return (h * (1.0 + self.weight.float())).type_as(x)
        return self.weight * h.to(x.dtype)

    @classmethod
    @torch.no_grad()
    def copy_of(cls, norm):
        from .common import norm_kind
        zc = norm_kind(norm) == "zero"
        new = cls(norm.weight.shape[-1], getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)), zc)
        new.to(norm.weight.device, norm.weight.dtype)
        new.weight.copy_(norm.weight)
        return new


def rope_blocks(x, cos, sin, n):
    """RoPE (rotate_half layout) on n consecutive rd-wide blocks of x (..., n * rd)."""
    rd = cos.shape[-1]
    h = rd // 2
    xs = x.view(*x.shape[:-1], n, rd)
    x1, x2 = xs[..., :h], xs[..., h:]
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return (torch.cat([x1, x2], -1) * c + torch.cat([-x2, x1], -1) * s).view(*x.shape)


def _apply_rope(x, cos, sin):
    """x (B, T, n, D): RoPE on the first rd = cos.shape[-1] dims of every head (partial rotary)."""
    rd = cos.shape[-1]
    c, s = cos.unsqueeze(2), sin.unsqueeze(2)
    xr, xp = x[..., :rd], x[..., rd:]
    h = rd // 2
    rot = torch.cat([-xr[..., h:], xr[..., :h]], -1)
    return torch.cat([xr * c + rot * s, xp], -1)


def _sdpa(Q, K, V, attention_mask, scaling):
    """Q (B, H, Tq, Dq), K (B, G, Tk, Dq), V (B, G, Tk, Dv); GQA expanded explicitly (the flash kernel caps the head
    dim at 256 and Mercurius' keys are wider; the memory-efficient kernel takes these shapes)."""
    rep = Q.shape[1] // K.shape[1]
    if rep > 1:
        K, V = K.repeat_interleave(rep, 1), V.repeat_interleave(rep, 1)
    if attention_mask is not None and attention_mask.dim() == 4:
        m = attention_mask[:, :, :, :K.shape[-2]]
        if m.dtype != torch.bool:
            m = m.to(Q.dtype)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=m, scale=scaling)
    tq, tk = Q.shape[-2], K.shape[-2]
    if tq > 1 and tk != tq:                                  # continuation on a cache: bottom-right causal mask
        m = torch.ones(tq, tk, dtype=torch.bool, device=Q.device).tril(tk - tq)
        return F.scaled_dot_product_attention(Q, K, V, attn_mask=m, scale=scaling)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=tq > 1, scale=scaling)


class MLAAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_groups, head_dim, rotary_dim, n_rope, rank, gated=True,
                 qk_eps=1e-6, zero_centered=True, k_rms_rank=None, per_head_q=True, layer_idx=0):
        super().__init__()
        H, G, D, rd = num_heads, num_kv_groups, head_dim, rotary_dim
        self.H, self.G, self.D, self.rd, self.n_rope, self.rank = H, G, D, rd, n_rope, rank
        self.gated, self.layer_idx, self.k_rms_rank = gated, layer_idx, k_rms_rank
        self.head_dim, self.scaling = D, D ** -0.5
        self.n_res = (G - n_rope) * rd
        self.q_proj = nn.Linear(hidden_size, H * D * (2 if gated else 1), bias=False)
        self.q_norm = RMSNorm(D, qk_eps, zero_centered)
        self.q_map = nn.Parameter(torch.eye(D).repeat(H, 1, 1)) if per_head_q else None
        self.k_rope = nn.Linear(hidden_size, n_rope * rd, bias=False)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up_k = nn.Linear(rank, self.n_res + G * (D - rd), bias=False)
        self.up_v = nn.Linear(rank, G * D, bias=False)
        self.k_rms = nn.Linear(hidden_size, G * (k_rms_rank or D), bias=False)
        self.k_eps = qk_eps
        self.register_buffer("rorope", torch.zeros(H, G, rd // 2))   # U_l[g(h), j] per head, frequency
        self.o_proj = nn.Linear(H * D, hidden_size, bias=False)
        self.kv_quant = None                                          # set by quant.qat (QAT / deployed cache)

    def spec(self):
        return {"kind": "mla", "num_heads": self.H, "num_kv_groups": self.G, "head_dim": self.D,
                "rotary_dim": self.rd, "n_rope": self.n_rope, "rank": self.rank, "gated": self.gated,
                "qk_eps": self.k_eps, "zero_centered": self.q_norm.zero_centered, "k_rms_rank": self.k_rms_rank,
                "per_head_q": self.q_map is not None}

    @classmethod
    def from_spec(cls, hidden_size, s, layer_idx):
        return cls(hidden_size, s["num_heads"], s["num_kv_groups"], s["head_dim"], s["rotary_dim"], s["n_rope"],
                   s["rank"], s["gated"], s["qk_eps"], s["zero_centered"], s["k_rms_rank"], s["per_head_q"],
                   layer_idx)

    def cache_values_per_token(self):
        return self.n_rope * self.rd + self.rank + self.G

    def inv_rms(self, x):
        """1 / RMS of each group's ORIGINAL key (k_norm's normaliser): (B, T, G, 1)."""
        y = self.k_rms(x).float()
        y = y.view(*x.shape[:-1], self.G, -1)
        ms = y.pow(2).sum(-1, keepdim=True) / self.D
        r = torch.rsqrt(ms + self.k_eps)
        if self.kv_quant is not None:
            r = r.half().float()                                      # the cache stores it in fp16
        return r

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        B, T = hidden_states.shape[:2]
        H, G, D, rd, nr = self.H, self.G, self.D, self.rd, self.n_rope
        half, nres = rd // 2, self.n_res
        qg = self.q_proj(hidden_states).view(B, T, H, -1)
        if self.gated:
            q, gate = qg[..., :D], qg[..., D:].reshape(B, T, -1)
        else:
            q, gate = qg, None
        q = self.q_norm(q)
        if self.q_map is not None:
            q = torch.einsum("bthd,hde->bthe", q.float(), self.q_map.float()).to(q.dtype)
        inv_r = self.inv_rms(hidden_states)
        cos, sin = position_embeddings
        cos, sin = cos[..., :rd], sin[..., :rd]
        kq = self.kv_quant
        k_rope = rope_blocks(self.k_rope(hidden_states).view(B, T, 1, nr * rd), cos.unsqueeze(2), sin.unsqueeze(2), nr)
        c = self.down(hidden_states)
        if kq is not None:
            k_rope = kq(k_rope.view(B, T, nr * rd), "k").view(B, T, 1, nr * rd)
            c = kq(c, "kv")
        kres = self.up_k(c)
        v = self.up_v(c)
        k_res, k_nope = kres[..., :nres], kres[..., nres:].view(B, T, G, D - rd)
        U = self.rorope.to(q.dtype)                                   # (H, G, half)
        qc = torch.cat([q[..., :half].unsqueeze(-2) * U, q[..., half:rd].unsqueeze(-2) * U], -1)   # (B,T,H,G,rd)
        q_rope = rope_blocks(qc[..., :nr, :].reshape(B, T, H, nr * rd), cos.unsqueeze(2), sin.unsqueeze(2), nr)
        Q = torch.cat([q_rope, qc[..., nr:, :].reshape(B, T, H, nres), q[..., rd:]], -1)
        K = torch.cat([k_rope.expand(B, T, G, nr * rd), k_res.view(B, T, 1, nres).expand(B, T, G, nres), k_nope], -1)
        K = (K.float() * inv_r).to(Q.dtype)
        V = v.view(B, T, G, D)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        if past_key_values is not None:
            K, V = past_key_values.update(K, V, self.layer_idx)
        out = _sdpa(Q, K, V, attention_mask, self.scaling).transpose(1, 2).reshape(B, T, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out), None


class PlainAttention(nn.Module):
    """GQA attention (optionally output-gated) with q/k norms and partial RoPE, plus the kv_quant hook."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, gated=True, qk_eps=1e-6, zero_centered=True,
                 layer_idx=0):
        super().__init__()
        self.H, self.G, self.D, self.gated, self.layer_idx = num_heads, num_kv_heads, head_dim, gated, layer_idx
        self.head_dim, self.scaling = head_dim, head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * (2 if gated else 1), bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, qk_eps, zero_centered)
        self.k_norm = RMSNorm(head_dim, qk_eps, zero_centered)
        self.kv_quant = None

    def spec(self):
        return {"kind": "plain", "num_heads": self.H, "num_kv_heads": self.G, "head_dim": self.D,
                "gated": self.gated, "qk_eps": self.q_norm.eps, "zero_centered": self.q_norm.zero_centered}

    @classmethod
    def from_spec(cls, hidden_size, s, layer_idx):
        return cls(hidden_size, s["num_heads"], s["num_kv_heads"], s["head_dim"], s["gated"], s["qk_eps"],
                   s["zero_centered"], layer_idx)

    def cache_values_per_token(self):
        return 2 * self.G * self.D

    @classmethod
    @torch.no_grad()
    def from_hf(cls, sa, layer_idx):
        H = sa.config.num_attention_heads
        D = sa.head_dim
        G = sa.config.num_key_value_heads
        gated = sa.q_proj.weight.shape[0] == 2 * H * D
        new = cls(sa.q_proj.weight.shape[1], H, G, D, gated, getattr(sa.q_norm, "eps",
                  getattr(sa.q_norm, "variance_epsilon", 1e-6)), True, layer_idx)
        new.q_norm, new.k_norm = RMSNorm.copy_of(sa.q_norm), RMSNorm.copy_of(sa.k_norm)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(new, n, getattr(sa, n))
        return new

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        B, T = hidden_states.shape[:2]
        H, G, D = self.H, self.G, self.D
        qg = self.q_proj(hidden_states).view(B, T, H, -1)
        q, gate = (qg[..., :D], qg[..., D:].reshape(B, T, -1)) if self.gated else (qg, None)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(hidden_states).view(B, T, G, D))
        v = self.v_proj(hidden_states).view(B, T, G, D)
        cos, sin = position_embeddings
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if self.kv_quant is not None:
            k = self.kv_quant(k.reshape(B, T, G * D), "k").view(B, T, G, D)
            v = self.kv_quant(v.reshape(B, T, G * D), "v").view(B, T, G, D)
        Q, K, V = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if past_key_values is not None:
            K, V = past_key_values.update(K, V, self.layer_idx)
        out = _sdpa(Q, K, V, attention_mask, self.scaling).transpose(1, 2).reshape(B, T, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return self.o_proj(out), None
