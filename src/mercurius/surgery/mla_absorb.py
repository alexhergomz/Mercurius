"""Absorbed MLA: attention computed directly on the cached latent.

The trainer and every evaluation run the latent layer UNABSORBED: c = down(x),
then K = k_norm(up_k c) and V = up_v c are materialised at full width and the
stock attention runs on them. Deployment wants the absorbed form, where only c
is cached and up_k / up_v are folded into the query and output side. The two
must be the same function or every number measured unabsorbed is biased
relative to what ships. This module is the absorbed form, written so that
equality can be checked rather than assumed (experiments/test_mla_absorb.py).

WHY ABSORPTION IS NOT THE TEXTBOOK IDENTITY HERE. Qwen3.5 applies k_norm, a
per-head RMSNorm, to the key AFTER k_proj:

    k_g = (1 + w_k) * (W_g c) / rms(W_g c),   rms(u) = sqrt(mean(u^2) + eps)

so K is not linear in c and q^T W_g c alone is wrong. The norm is still
computable from the latent, because

    mean((W_g c)^2) = c^T (W_g^T W_g) c / d = c^T M_g c / d

with M_g a d_c x d_c matrix fixed at conversion time. So the absorbed score is

    score_h(t, s) = [ (W_g^T ((1 + w_k) * q~_h))^T c_s ] * rsqrt(c_s^T M_g c_s / d + eps)

exact in exact arithmetic. The per-token scalar r_g(s) = rsqrt(...) can be
cached alongside c (n_kv floats per token, 4 against d_c = 512) or recomputed.

Two further conditions, both met by this pipeline and checked by the test:

  * NO ROTARY on the absorbed dims. RoPE rotates k AFTER up_k, and a
    position-dependent rotation between W_g and the query breaks the fold --
    which is why DeepSeek needs a decoupled RoPE key. Stage C dials all 32
    frequencies to identity (NoPE), so there is nothing to decouple. Any
    partial dial (c0/c1) makes this module refuse.
  * The OUTPUT GATE is elementwise on head_dim, between up_v and o_proj, so
    up_v cannot be folded into o_proj. It does not need to be: attention is
    computed on the latent, P_h c, and up-projected once per head per query,
    which is still the absorbed cost profile (nothing per-key at full width).

Per-head query maps (PerHeadQ) sit inside q_proj, before q_norm, so they fold
into the query like any other query-side weight.
"""
import torch
import torch.nn.functional as F


def _latent(sa):
    lat = getattr(sa.k_proj, "latent", None)
    if lat is None:
        raise ValueError("self_attn has no latent KV (run convert_to_mla first)")
    if getattr(lat, "blend", False):
        raise ValueError("blended latent is not absorbable; merge the blend first")
    return lat


def _check_nope(model):
    from mercurius.surgery.norm_fusion import get_trunk
    d = getattr(get_trunk(model).rotary_emb, "_dial", None)
    if d is None or d["keep_freqs"] != 0:
        raise ValueError("absorption is exact only under full NoPE (stage C "
                         "with keep 0); rotary between up_k and the query breaks it")


@torch.no_grad()
def precompute(sa, n_kv, head_dim):
    """Per-group key maps W_g (d x d_c) and norm metrics M_g = W_g^T W_g."""
    lat = _latent(sa)
    Wk = lat.up_k.weight.float()                        # (n_kv*d, d_c)
    Wg = Wk.view(n_kv, head_dim, -1)                    # (n_kv, d, d_c)
    Mg = Wg.transpose(1, 2) @ Wg                        # (n_kv, d_c, d_c)
    Wv = lat.up_v.weight.float().view(n_kv, head_dim, -1)
    return Wg, Mg, Wv


def absorbed_attention(sa, x, n_heads, n_kv, head_dim, cache=None,
                       dtype=torch.float32):
    """Attention output (before the residual add) for hidden states x (B,T,D),
    causal, computed on the latent. `cache` holds the precompute() tuple.

    Returns (out, c, r): c (B,T,d_c) and r (B,T,n_kv) are exactly what an
    absorbed KV cache would store.
    """
    lat = _latent(sa)
    Wg, Mg, Wv = cache if cache is not None else precompute(sa, n_kv, head_dim)
    B, T, _ = x.shape
    rep = n_heads // n_kv

    qg = sa.q_proj(x).view(B, T, n_heads, 2 * head_dim)
    q, gate = qg.split(head_dim, dim=-1)
    q = sa.q_norm(q).to(dtype)                          # (B,T,H,d); NoPE: no rotation
    c = lat.down(x).to(dtype)                           # (B,T,d_c) -- the cache

    eps = sa.k_norm.eps
    gk = (1.0 + sa.k_norm.weight.to(dtype))             # (d,)
    # per-token key normaliser, from the latent alone
    quad = torch.einsum("btc,gce,bte->btg", c, Mg.to(dtype), c)
    r = torch.rsqrt(quad / head_dim + eps)              # (B,T,n_kv)

    # absorbed queries: q_abs = W_g^T (gk * q~), one per query head
    Wg_h = Wg.to(dtype).repeat_interleave(rep, dim=0)   # (H, d, d_c)
    q_abs = torch.einsum("bthd,hdc->bthc", q * gk, Wg_h)  # (B,T,H,d_c)

    r_h = r.repeat_interleave(rep, dim=-1)              # (B,T,H)
    scores = torch.einsum("bthc,bsc->bhts", q_abs, c)
    scores = scores * r_h.permute(0, 2, 1).unsqueeze(2) * sa.scaling
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    P = scores.softmax(-1)                              # (B,H,T,T)

    o_lat = torch.einsum("bhts,bsc->bthc", P, c)        # attention ON THE LATENT
    Wv_h = Wv.to(dtype).repeat_interleave(rep, dim=0)   # (H, d, d_c)
    o = torch.einsum("bthc,hdc->bthd", o_lat, Wv_h)     # up_v once per query
    o = o.reshape(B, T, -1) * torch.sigmoid(gate.reshape(B, T, -1).to(dtype))
    out = sa.o_proj(o.to(x.dtype))
    return out, c, r


def install_absorbed(model, dtype=torch.float32):
    """Swap every latent attention layer's forward for the absorbed one.
    Prefill only (no incremental cache): this exists to measure equivalence.
    Returns a restore function."""
    from mercurius.surgery.norm_fusion import get_trunk
    _check_nope(model)
    cfg = getattr(model.config, "text_config", model.config)
    H, G, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    saved = []
    for l in get_trunk(model).layers:
        sa = getattr(l, "self_attn", None)
        if sa is None or getattr(sa.k_proj, "latent", None) is None:
            continue
        pre = precompute(sa, G, d)
        orig = sa.forward

        def fwd(hidden_states, *args, _sa=sa, _pre=pre, **kw):
            out, _, _ = absorbed_attention(_sa, hidden_states, H, G, d, _pre, dtype)
            return out, None
        sa.forward = fwd
        saved.append((sa, orig))

    def restore():
        for sa, orig in saved:
            sa.forward = orig
    return restore
