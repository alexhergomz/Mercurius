"""GQA attention -> absorbable MLA (arch/attention.MLAAttention), calibrated on the training mix.

Per attention layer (G KV heads, head_dim D, rd rotary dims = rd/2 frequency pairs (i, i + rd/2)):

  k_norm   the key is k_norm(W_k x) = gain * (W_k x) / rms_g(W_k x): the per-dim gain is folded into every key row,
           the per-group 1/rms is kept exact (k_rms = the frozen original W_k, or a low-rank estimate).
  RoRoPE   (TransMLA) per frequency l, the G heads' gained (re, im) key components are mixed by one orthogonal
           U_l = principal axes of Wre_l C Wre_l^T + Wim_l C Wim_l^T (C: input covariance). The same U_l acts on
           re and im, so it commutes with RoPE; component 0 carries the most key energy at that frequency.
  RoPE key components 0 .. n_rope-1: an exact linear map of x, rotated at its own position, cached.
  latent   components n_rope .. G-1 lose RoPE and join every head's gained NoPE rows and W_v in ONE CARE latent:
           SVD of the K/V-balanced [W_k'; W_v] in the input-whitened basis (minimises activation error, not weight
           error), truncated at the layer's rank.
  ranks    water-filled across layers on the whitened spectra under a total budget, IN MULTIPLES of
           `rank_multiple` (Mercurius-1 allocated one rank at a time and ended at 579, 451, 390, ...: no block format
           could store the up-projections and cuBLAS left its tensor-core paths, docs/porting_findings.md #3, #15).
  sequence layers are converted in depth order; each layer's input covariance is re-collected from the model with
           every earlier layer already compressed (SVD-LLM / GPTQ-style sequential calibration).

Exactness: with n_rope = G and full rank the converted layer reproduces the original (tested).
"""
from __future__ import annotations

import heapq
import math

import torch

from mercurius.arch.attention import MLAAttention, RMSNorm
from mercurius.arch.common import norm_gain, rotary_dim, text_config, trunk


def whiten_factor(cov, eps=1e-6):
    """L with cov = L L^T (Cholesky, jittered if near-singular; symmetric root as a last resort)."""
    d = cov.shape[0]
    jitter = eps * torch.diag(cov).mean().clamp_min(1e-12)
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    for _ in range(6):
        try:
            return torch.linalg.cholesky(cov + jitter * eye)
        except Exception:
            jitter *= 10
    w, V = torch.linalg.eigh(cov)
    return V @ torch.diag(w.clamp_min(0).sqrt()) @ V.T


@torch.no_grad()
def rorope_split(Wk, gain, cov, G, D, rd, n_rope):
    """Wk (G*D, d) original key rows, gain (D,), cov (d, d). Returns U (rd/2, G, G), the RoPE-key rows
    (n_rope*rd, d) and the latent's key rows ((G-n_rope)*rd + G*(D-rd), d), fp64."""
    half = rd // 2
    W = Wk.double().view(G, D, -1) * gain.double().view(1, D, 1)
    S = cov.double()
    U = torch.empty(half, G, G, dtype=torch.float64, device=W.device)
    for l in range(half):
        Wre, Wim = W[:, l], W[:, l + half]
        C = Wre @ S @ Wre.T + Wim @ S @ Wim.T
        ev, V = torch.linalg.eigh(C)
        V = V[:, torch.argsort(ev, descending=True)]
        idx = V.abs().argmax(0)                                # deterministic sign
        U[l] = V * torch.sign(V[idx, torch.arange(G)]).unsqueeze(0)
    comps = []
    for j in range(G):
        re = torch.stack([U[l][:, j] @ W[:, l] for l in range(half)])
        im = torch.stack([U[l][:, j] @ W[:, l + half] for l in range(half)])
        comps.append(torch.cat([re, im], 0))
    rope_rows = torch.cat(comps[:n_rope], 0)
    k_rows = torch.cat([comps[j] for j in range(n_rope, G)] + [W[g, rd:] for g in range(G)], 0)
    return U, rope_rows, k_rows


def _balanced(Wk, Wv):
    sk = Wk.norm() / math.sqrt(Wk.numel())
    sv = Wv.norm() / math.sqrt(Wv.numel())
    g = (sk * sv).sqrt()
    return float(g / sk), float(g / sv)


@torch.no_grad()
def care_factor(Wk, Wv, cov, r):
    """CARE: rank-r factorisation of [Wk; Wv] minimising activation error under input covariance cov.
    Returns (down (r, d), up_k (k_out, r), up_v (v_out, r), energy kept), fp32."""
    Wk, Wv = Wk.double(), Wv.double()
    ks, vs = _balanced(Wk, Wv)
    W = torch.cat([Wk * ks, Wv * vs], 0)
    L = whiten_factor(cov.double().to(W.device))
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    r = min(r, S.numel())
    Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    up = U[:, :r] * S[:r]
    down = Vh[:r] @ Linv
    k_out = Wk.shape[0]
    energy = float(S[:r].pow(2).sum() / S.pow(2).sum())
    return down.float(), (up[:k_out] / ks).float(), (up[k_out:] / vs).float(), energy


@torch.no_grad()
def whitened_spectrum(sa, cov, n_rope, cfg):
    """Singular values of the whitened operator the latent truncates (for water-filling)."""
    G, D, rd = cfg.num_key_value_heads, cfg.head_dim, rotary_dim(cfg)
    _, _, Kr = rorope_split(sa.k_proj.weight.float(), norm_gain(sa.k_norm), cov, G, D, rd, n_rope)
    Wv = sa.v_proj.weight.double()
    ks, vs = _balanced(Kr, Wv)
    M = torch.cat([Kr * ks, Wv * vs], 0)
    return torch.linalg.svdvals(M @ whiten_factor(cov.double().to(M.device)))


def allocate_ranks(spectra, budget, step=128, min_rank=128):
    """Greedy water-filling in units of `step`: every layer starts at min_rank, then the next `step` ranks go to
    the layer whose next `step` squared singular values (the activation error they remove) are largest."""
    if budget % step or min_rank % step:
        raise ValueError("budget and min_rank must be multiples of step")
    ranks = {k: min_rank for k in spectra}
    used = sum(ranks.values())
    if used > budget:
        raise ValueError(f"budget {budget} below {len(spectra)} x min_rank {min_rank}")

    def gain(k, r):
        s2 = spectra[k].double().pow(2)
        if r + step > s2.numel():
            return -1.0
        return float(s2[r:r + step].sum())

    heap = [(-gain(k, ranks[k]), k) for k in spectra]
    heapq.heapify(heap)
    while used + step <= budget and heap:
        g, k = heapq.heappop(heap)
        if -g <= 0:
            continue
        ranks[k] += step
        used += step
        heapq.heappush(heap, (-gain(k, ranks[k]), k))
    return ranks


@torch.no_grad()
def lowrank_rms_rows(Wk, G, D, rank, cov):
    """Rows P_g (rank, d) with ||P_g x||^2 ~= ||W_g x||^2 under the input distribution: top eigenvectors of
    C^1/2 W_g^T W_g C^1/2, mapped back. Replaces the full G*D-row key-RMS projection by G*rank rows
    (docs/porting_findings.md #4). rank = None keeps it exact."""
    L = whiten_factor(cov.double().to(Wk.device))
    rows = []
    for g in range(G):
        Wg = Wk[g * D:(g + 1) * D].double()
        M = Wg @ L                                             # E||W_g x||^2 = ||W_g L||_F^2
        _, S, Vh = torch.linalg.svd(M, full_matrices=False)
        Linv = torch.linalg.solve_triangular(L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
        rows.append(S[:rank, None] * (Vh[:rank] @ Linv))
    return torch.cat(rows, 0).float()


@torch.no_grad()
def convert_layer(sa, cov, rank, n_rope, cfg, layer_idx, per_head_q=True, k_rms_rank=None):
    """HF attention module -> MLAAttention (same device / dtype)."""
    H, G, D, rd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, rotary_dim(cfg)
    if not 1 <= n_rope <= G:
        raise ValueError(f"n_rope must be in 1..{G}")
    Wq = sa.q_proj.weight
    gated = Wq.shape[0] == 2 * H * D
    dev, dt = Wq.device, Wq.dtype
    Wk = sa.k_proj.weight.float()
    U, rope_rows, k_rows = rorope_split(Wk, norm_gain(sa.k_norm), cov.to(dev), G, D, rd, n_rope)
    down, up_k, up_v, energy = care_factor(k_rows, sa.v_proj.weight.float(), cov.to(dev), rank)
    rank = down.shape[0]
    qn = RMSNorm.copy_of(sa.q_norm)
    new = MLAAttention(Wq.shape[1], H, G, D, rd, n_rope, rank, gated, qn.eps, qn.zero_centered, k_rms_rank,
                       per_head_q, layer_idx).to(dev, dt)
    new.q_norm = qn
    new.q_proj.weight.copy_(Wq)
    new.o_proj.weight.copy_(sa.o_proj.weight)
    new.k_rope.weight.copy_(rope_rows.to(dt))
    new.down.weight.copy_(down.to(dt))
    new.up_k.weight.copy_(up_k.to(dt))
    new.up_v.weight.copy_(up_v.to(dt))
    new.k_rms.weight.copy_((Wk if not k_rms_rank else lowrank_rms_rows(Wk, G, D, k_rms_rank, cov.to(dev))).to(dt))
    new.k_rms.weight.requires_grad_(False)
    new.k_eps = getattr(sa.k_norm, "eps", getattr(sa.k_norm, "variance_epsilon", 1e-6))
    g_of_h = torch.arange(H, device=dev) // (H // G)
    new.rorope.copy_(U.permute(1, 2, 0)[g_of_h].float())
    if new.q_map is not None:
        new.q_map.data = new.q_map.data.float()
    return new, energy


@torch.no_grad()
def collect_covs(model, windows, layers, device):
    """Input covariance of k_proj at each trunk layer in `layers` (fp64 accumulation, one forward per window)."""
    tr = trunk(model)
    acc, cnt = {i: None for i in layers}, {i: 0 for i in layers}

    def hook(i):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            acc[i] = x.t() @ x if acc[i] is None else acc[i] + x.t() @ x
            cnt[i] += x.shape[0]
        return h
    hs = [tr.layers[i].self_attn.k_proj.register_forward_hook(hook(i)) for i in layers]
    was = model.training
    model.eval()
    try:
        for w in windows:
            model(input_ids=w.reshape(1, -1).to(device), logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
        model.train(was)
    return {i: (acc[i] / max(cnt[i], 1)).float() for i in layers}


@torch.no_grad()
def convert_model(model, windows, budget, n_rope=1, rank_multiple=128, min_rank=128, per_head_q=True,
                  k_rms_rank=None, device="cuda", log=print):
    """Sequential CARE calibration + conversion of every full-attention layer. Returns {layer: (rank, energy)}."""
    cfg = text_config(model)
    tr = trunk(model)
    attn = [i for i, l in enumerate(tr.layers) if hasattr(l, "self_attn") and hasattr(l.self_attn, "k_proj")]
    if not attn:
        return {}
    covs0 = collect_covs(model, windows, attn, device)
    spectra = {i: whitened_spectrum(tr.layers[i].self_attn, covs0[i].to(device), n_rope, cfg) for i in attn}
    alloc = allocate_ranks(spectra, budget, rank_multiple, min_rank)
    log(f"  MLA: {len(windows)} calibration windows; ranks (multiples of {rank_multiple}, budget {budget}): "
        f"{dict(sorted(alloc.items()))}")
    info = {}
    for j, i in enumerate(attn):
        cov = covs0[i] if j == 0 else collect_covs(model, windows, [i], device)[i]
        layer = tr.layers[i]
        layer.self_attn, e = convert_layer(layer.self_attn, cov.to(device), alloc[i], n_rope, cfg, i, per_head_q,
                                           k_rms_rank)
        info[i] = (alloc[i], e)
        log(f"    layer {i:>2}: rank {alloc[i]:>4}  energy kept {e:.4f}")
    G, rd = cfg.num_key_value_heads, rotary_dim(cfg)
    per_tok = sum(r for r, _ in info.values()) + len(info) * (n_rope * rd + G)
    log(f"  MLA cache: {per_tok} values/token over {len(info)} layers "
        f"({len(info) * 2 * G * cfg.head_dim / per_tok:.2f}x smaller than GQA)")
    return info


@torch.no_grad()
def project_rope_commute_(model, rd):
    """Keep every per-head query map in the RoPE commutant: per rotary pair (i, i + rd/2) a 2x2 [[a, -b], [b, a]],
    no rotary <-> NoPE mixing; the NoPE block is free. Idempotent; identity stays identity."""
    half = rd // 2
    i = torch.arange(half)
    j = i + half
    for m in model.modules():
        if isinstance(m, MLAAttention) and m.q_map is not None:
            R = m.q_map.data
            a = (R[:, i, i] + R[:, j, j]) / 2
            b = (R[:, i, j] - R[:, j, i]) / 2
            R[:, :rd, :] = 0
            R[:, :, :rd] = 0
            R[:, i, i] = a
            R[:, j, j] = a
            R[:, i, j] = b
            R[:, j, i] = -b


@torch.no_grad()
def tie_rope_norm_deltas_(model, rd, ref):
    """A per-dim q_norm gain commutes with the pair rotation only if both dims of a pair share it: tie the TRAINED
    change of each gain per rotary pair (the original gains are kept)."""
    half = rd // 2
    for name, p in model.named_parameters():
        if name in ref:
            d = p.data[:rd] - ref[name][:rd]
            avg = (d[:half] + d[half:]) / 2
            p.data[:half] = ref[name][:half] + avg
            p.data[half:rd] = ref[name][half:rd] + avg
