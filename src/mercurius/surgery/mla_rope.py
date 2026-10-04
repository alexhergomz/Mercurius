"""Absorbable MLA: decoupled RoPE key via RoRoPE (TransMLA), CARE latent for the rest (#68).

c0 (transmla.py) compresses the PRE-RoPE key -- rotary dims included -- into the latent and
rotates the reconstruction, so decode must rebuild every cached token's rotary key: not
absorbable (~9x the attention FLOPs of absorbed MLA at long context). This module does what
DeepSeek-V2 / TransMLA do instead: a small RoPE key OUTSIDE the latent, cached exactly, and
everything else in the latent, where W_UK folds into the query.

Per attention layer (G = 4 KV heads, head_dim 256, rotary dims 64 = 32 pairs (i, i+32)):

  k_norm.  The stock key is k_norm(W_k x) = (1 + w) * (W_k x) / rms_g(W_k x), per KV head g.
           The per-dim gain D = 1 + w is FOLDED into every key row below; the per-head,
           per-token scalar 1/rms_g is computed EXACTLY from the frozen original W_k and
           cached (G scalars per token per layer).
  RoRoPE.  (TransMLA Eq. 19) For each frequency l, the G heads' gained (re, im) components
           are rotated by one orthogonal U_l (G x G), the SAME for re and im -- it commutes
           with RoPE. U_l = principal axes of
               C_l = Wre_l Sigma Wre_l^T + Wim_l Sigma Wim_l^T     (Sigma: input covariance)
           so component 0 carries the most key energy at that frequency.
  RoPE key. Components 0 .. n_rope-1 (64 dims each, shared by all heads): an exact linear
           map of x (k_rope, trainable), rotated at its own position, cached.
  Latent.  Components n_rope .. G-1 lose RoPE (TransMLA: "remove their RoPE encoding while
           preserving positional information within the principal components") and join the
           gained NoPE rows of every head and W_v in ONE CARE latent: whitened SVD of
           [W_k'; W_v] under the same input covariance, water-filled ranks -- unchanged.
  Score.   head h in group g:  q_h . k_g / rms_g =
               sum_{j<n_rope}  (U_l[g,j] q_l) . Rot(Delta) kt_j     (exact RoPE)
             + sum_{j>=n_rope} (U_l[g,j] q_l) . kt_j                 (RoPE dropped)
             + q_nope . (D k_g)_nope
           all divided by rms_g. With n_rope = G it is EXACTLY the stock score (tested).
           Built as one GQA attention call on Q' = [rot(Uq) | Uq | q_nope] and
           K' = [rot(kt_rope) | kt_res | k_nope_g] / rms_g. Every K' block is a linear
           function of the cached (kt_rope, c, rms), so decode is absorbable by construction.
Cache per token per layer: n_rope*64 + r_layer + G.
"""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg(model):
    return model.config.text_config if hasattr(model.config, "text_config") else model.config


def _rot_dim(cfg):
    rp = getattr(cfg, "rope_parameters", None) or {}
    return int(cfg.head_dim * float(rp.get("partial_rotary_factor", 1.0)))


@torch.no_grad()
def rorope_split(Wk, gain, cov, G, D, rd, n_rope):
    """Wk (G*D, d) original key rows, gain (D,) = 1 + k_norm.weight, cov (d, d).
    Returns U (half, G, G) and the rows of the RoPE key (n_rope*rd, d) and of the
    latent's key part ((G-n_rope)*rd + G*(D-rd), d), all fp64."""
    half = rd // 2
    W = Wk.double().view(G, D, -1) * gain.double().view(1, D, 1)       # gained rows
    S = cov.double()
    U = torch.empty(half, G, G, dtype=torch.float64, device=W.device)
    comps = []                                                        # per j: (re, im) rows
    for l in range(half):
        Wre, Wim = W[:, l], W[:, l + half]                            # (G, d)
        C = Wre @ S @ Wre.T + Wim @ S @ Wim.T
        ev, V = torch.linalg.eigh(C)
        V = V[:, torch.argsort(ev, descending=True)]
        # deterministic sign: largest-|.| entry of each axis positive (replays must agree)
        idx = V.abs().argmax(0)
        V = V * torch.sign(V[idx, torch.arange(G)]).unsqueeze(0)
        U[l] = V
    for j in range(G):
        re = torch.stack([U[l][:, j] @ W[:, l] for l in range(half)])        # (half, d)
        im = torch.stack([U[l][:, j] @ W[:, l + half] for l in range(half)])
        comps.append(torch.cat([re, im], 0))                                # (rd, d)
    rope_rows = torch.cat(comps[:n_rope], 0)
    res_rows = [comps[j] for j in range(n_rope, G)]
    nope_rows = [W[g, rd:] for g in range(G)]
    k_rows = torch.cat(res_rows + nope_rows, 0)
    return U, rope_rows, k_rows


@torch.no_grad()
def decoupled_spectra(model, covs, n_rope):
    """Whitened singular values of the operator the latent truncates (cf. whitened_spectra),
    for water-filling the budget across layers."""
    from mercurius.surgery.norm_fusion import get_trunk
    from mercurius.surgery.transmla import whiten_factor, merged_weight
    cfg = _cfg(model)
    G, D, rd = cfg.num_key_value_heads, cfg.head_dim, _rot_dim(cfg)
    out = {}
    for i, l in enumerate(get_trunk(model).layers):
        sa = getattr(l, "self_attn", None)
        if sa is None or i not in covs:
            continue
        Wk, _ = merged_weight(sa.k_proj)
        Wv, _ = merged_weight(sa.v_proj)
        _, _, Kr = rorope_split(Wk, 1.0 + sa.k_norm.weight.float(), covs[i], G, D, rd, n_rope)
        Kr = Kr.float()
        sk = Kr.norm() / math.sqrt(Kr.numel())
        sv = Wv.float().norm() / math.sqrt(Wv.numel())
        g = (sk * sv).sqrt()
        M = torch.cat([Kr * (g / sk), Wv.float() * (g / sv)], 0)
        L = whiten_factor(covs[i].to(M.dtype).to(M.device))
        out[i] = torch.linalg.svdvals(M @ L)
    return out


def _rope_blocks(x, cos, sin, n):
    """Rotate n consecutive rd-dim blocks of x (..., n*rd) with RoPE (rotate_half layout)."""
    rd = cos.shape[-1]
    h = rd // 2
    xs = x.view(*x.shape[:-1], n, rd)
    x1, x2 = xs[..., :h], xs[..., h:]
    c, s = cos.unsqueeze(-2), sin.unsqueeze(-2)
    out = torch.cat([x1, x2], -1) * c + torch.cat([-x2, x1], -1) * s
    return out.view(*x.shape)


def _decoupled_forward(self, hidden_states, position_embeddings, attention_mask,
                       past_key_values=None, **kwargs):
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        ALL_ATTENTION_FUNCTIONS, eager_attention_forward)
    B, T = hidden_states.shape[:2]
    H, G, D = self._H, self._G, self.head_dim
    rd, nr = self._rd, self._n_rope
    half = rd // 2
    lat = self.k_proj.latent
    q, gate = torch.chunk(self.q_proj(hidden_states).view(B, T, H, 2 * D), 2, dim=-1)
    gate = gate.reshape(B, T, -1)
    q = self.q_norm(q)                                     # (B,T,H,D), post-norm map inside
    # exact per-head 1/rms of the ORIGINAL key (k_norm's normalisation), cached at decode
    kraw = self.k_rms(hidden_states).view(B, T, G, D).float()
    inv_r = torch.rsqrt(kraw.pow(2).mean(-1, keepdim=True)
                        + getattr(self.k_norm, "eps", 1e-6))           # (B,T,G,1)
    Wr = lat.k_rope.weight
    if getattr(self, "_qat", None) is not None:
        from mercurius.models.qat import nf4_roundtrip, ste, tq_roundtrip
        Wr = ste(Wr, nf4_roundtrip(Wr))
    kt0 = F.linear(hidden_states, Wr.to(hidden_states.dtype))       # (B,T,nr*rd)
    if getattr(self, "_qat", None) is not None and self._qat["kv_bits"] < 16:
        # the cached RoPE key gets the latent's quantizer. TurboQuant needs its random
        # rotation in front (data-oblivious codebook): a fixed orthogonal R0 per layer,
        # cache = Q(R0 kt), read = R0^T deq -- RoPE is applied after, at read time.
        if self._qat.get("kv_quant", "int") == "tq":
            from mercurius.models.qat import _orth
            if getattr(self, "_qat_R0", None) is None:
                self._qat_R0 = _orth(kt0.shape[-1], 7000 + int(self.layer_idx)).to(kt0.device)
            R0 = self._qat_R0.to(kt0.dtype)
            kt0 = ste(kt0, tq_roundtrip(kt0 @ R0.T, self._qat["kv_bits"]) @ R0)
        else:
            from mercurius.models.qat import int_group_roundtrip
            kt0 = ste(kt0, int_group_roundtrip(kt0, self._qat["kv_bits"],
                                               self._qat["kv_group"]))
    kres, v = lat(hidden_states)                           # (B,T,(G-nr)*rd + G*(D-rd)), (B,T,G*D)
    nres = (G - nr) * rd
    k_res, k_nope = kres[..., :nres], kres[..., nres:].view(B, T, G, D - rd)
    # query side: per head, per frequency, the coefficients U_l[g(h), j]
    Ug = self._U_heads.to(q.dtype)                         # (H, G, half): U_l[g(h), j]
    qre, qim = q[..., :half], q[..., half:rd]              # (B,T,H,half)
    qc = torch.cat([qre.unsqueeze(-2) * Ug, qim.unsqueeze(-2) * Ug], -1)  # (B,T,H,G,rd)
    cos, sin = position_embeddings
    q_rope = _rope_blocks(qc[..., :nr, :].reshape(B, T, H, nr * rd),
                          cos.unsqueeze(2), sin.unsqueeze(2), nr)
    q_res = qc[..., nr:, :].reshape(B, T, H, nres)
    Q = torch.cat([q_rope, q_res, q[..., rd:]], -1)        # (B,T,H,nr*rd + nres + D-rd)
    k_rope = _rope_blocks(kt0.view(B, T, 1, nr * rd), cos.unsqueeze(2), sin.unsqueeze(2), nr)
    K = torch.cat([k_rope.expand(B, T, G, nr * rd),
                   k_res.view(B, T, 1, nres).expand(B, T, G, nres), k_nope], -1)
    K = (K.float() * inv_r).to(Q.dtype)
    V = v.view(B, T, G, D)
    Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
    if past_key_values is not None:
        K, V = past_key_values.update(K, V, self.layer_idx)
    # Direct SDPA, KV heads expanded explicitly. Through transformers' wrapper the call
    # carries enable_gqa=True, which only the FLASH kernel takes -- and flash caps the head
    # dim at 256 (ours: 448 q/k, 256 v) -- so it fell back to the MATH kernel and
    # materialised the full score matrix (64 GiB at 32k, OOM at step ~110). The
    # memory-efficient kernel takes these shapes with a causal flag or a mask.
    rep = H // G
    K = K.repeat_interleave(rep, dim=1)
    V = V.repeat_interleave(rep, dim=1)
    if attention_mask is not None:
        m = attention_mask[:, :, :, :K.shape[-2]]
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=m, scale=self.scaling)
    elif Q.shape[-2] > 1 and K.shape[-2] != Q.shape[-2]:
        # continuation on a cache: is_causal would align top-left -- build bottom-right
        tq, tk = Q.shape[-2], K.shape[-2]
        m = torch.ones(tq, tk, dtype=torch.bool, device=Q.device).tril(tk - tq)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=m, scale=self.scaling)
    else:
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=Q.shape[-2] > 1,
                                             scale=self.scaling)
    out = out.transpose(1, 2).reshape(B, T, -1).contiguous() * torch.sigmoid(gate)
    return self.o_proj(out), None


@torch.no_grad()
def convert_layer_decoupled(sa, cov, rank, n_rope, cfg):
    """Replace sa's k/v with [RoPE key + CARE latent], keep the forward exact-by-design."""
    from mercurius.surgery.transmla import LatentKV, FactoredKProj, merged_weight
    G, D, H, rd = cfg.num_key_value_heads, cfg.head_dim, cfg.num_attention_heads, _rot_dim(cfg)
    k_orig, v_proj = sa.k_proj, sa.v_proj
    Wk, _ = merged_weight(k_orig)
    gain = 1.0 + sa.k_norm.weight.float()
    U, rope_rows, k_rows = rorope_split(Wk, gain, cov, G, D, rd, n_rope)
    dev, dt = Wk.device, k_orig.weight.dtype
    k_lin = nn.Linear(Wk.shape[1], k_rows.shape[0], bias=False).to(dev, dt)
    k_lin.weight.copy_(k_rows.to(dt))
    lat = LatentKV(k_lin, v_proj, rank, cov=cov)
    lat.k_rope = nn.Linear(Wk.shape[1], rope_rows.shape[0], bias=False).to(dev, dt)
    lat.k_rope.weight.copy_(rope_rows.to(dt))
    sa.k_rms = k_orig                                       # frozen, for the exact 1/rms
    for p in sa.k_rms.parameters():
        p.requires_grad_(False)
    sa.k_proj = FactoredKProj(lat, "k")
    sa.v_proj = FactoredKProj(lat, "v")
    hpg = H // G
    g_of_h = torch.arange(H) // hpg
    # (H, G, half): coefficient of component j for head h at frequency l = U_l[g(h), j]
    sa._U_heads = U.permute(1, 2, 0)[g_of_h].float().to(dev)
    sa._U = U.float().to(dev)
    sa._H, sa._G, sa._rd, sa._n_rope = H, G, rd, n_rope
    sa._rope_decoupled = True
    sa.forward = types.MethodType(_decoupled_forward, sa)
    return rank, lat.full_rank, lat.energy_kept


def convert_to_mla_decoupled(model, alloc, covs, n_rope=1, only=None, verbose=True):
    from mercurius.surgery.norm_fusion import get_trunk
    cfg = _cfg(model)
    info = []
    for i, l in enumerate(get_trunk(model).layers):
        sa = getattr(l, "self_attn", None)
        if sa is None or getattr(sa, "_rope_decoupled", False):
            continue
        if only is not None and i not in only:
            continue
        r, fr, e = convert_layer_decoupled(sa, covs[i], int(alloc[i]), n_rope, cfg)
        info.append((i, r, fr, e))
    if verbose and info:
        rd = _rot_dim(cfg)
        tot = sum(r for _, r, _, _ in info) + len(info) * (n_rope * rd + cfg.num_key_value_heads)
        full = len(info) * 2 * cfg.num_key_value_heads * cfg.head_dim
        print(f"  absorbable MLA (decoupled RoPE, RoRoPE n_rope={n_rope}) on {len(info)} "
              f"layers: latent {sum(r for _, r, _, _ in info)} + RoPE key "
              f"{len(info) * n_rope * rd} + rms {len(info) * cfg.num_key_value_heads} = "
              f"{tot} values/token ({full / tot:.2f}x vs {full})", flush=True)
    return info
