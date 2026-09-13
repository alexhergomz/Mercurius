"""Stage D — TransMLA: joint low-rank latent for K and V on the attention layers.

Half of TransMLA is already done. Its two parts are (1) RoRoPE/FreqFold, which
concentrates positional information and strips RoPE from the remaining heads,
and (2) balanced low-rank compression of the KV latent. Part (1) IS the dial in
stage_c_dial.py, already run to full NoPE and recovered to +0.65%. What remains
is only the compression.

Measured payoff on this model: 6 of 24 layers hold a KV cache, 12.0 KiB/token
today (a dense 24-layer transformer would hold 48.0). A latent of d_c=512 halves
that, d_c=256 quarters it. Modest -- the hybrid already removed 4x -- so this is
architectural alignment more than a memory win.

TWO THINGS THAT MAKE THIS SAFE HERE:

  * We never touch q_proj. Qwen3.5 sets attn_output_gate=true, so q_proj is
    (4096, 1024) where only the first 2048 rows are Q and the rest is the fused
    output gate; naive head-merging corrupts it. Compressing only K and V
    sidesteps that entirely -- the gate is never in the factorization.

  * Balanced scaling before SVD. K and V have different magnitudes, and a joint
    SVD would otherwise spend its rank budget on whichever is larger. TransMLA
    calls this KV balancing; without it the truncation is silently lopsided.

Per-layer adaptive rank follows YouZhi's finding that degradation varies by
layer and a single global size leaves quality on the table -- here via spectral
energy rather than a fixed d_c.
"""
import math
import torch
import torch.nn as nn


def merged_weight(mod):
    """Effective weight of a projection, folding any LoRA delta in.

    After inject_lora, k_proj/v_proj are LoRALinear wrappers with no .weight --
    reading mod.weight would raise. And the trained delta is part of the learned
    mapping, so factorizing the frozen base alone would silently discard
    everything training accomplished on these layers.
    """
    if hasattr(mod, "base"):                       # LoRALinear
        W = mod.base.weight.data.float()
        if getattr(mod, "lora_A", None) is not None:
            W = W + mod.scale * (mod.lora_B.data.float() @ mod.lora_A.data.float())
        return W, mod.base
    return mod.weight.data.float(), mod


class LatentKV(nn.Module):
    """Replaces k_proj and v_proj with a shared low-rank latent.

        c = W_dkv x            (d_c)        <- this is what the cache stores
        K = W_uk c             (n_kv*hd)
        V = W_uv c             (n_kv*hd)

    Exact when d_c == rank([W_k; W_v]); lossy below that.
    """

    @staticmethod
    def v_output_metric(o_proj, n_kv, head_dim):
        """M_V[g] = sum_{h in group g} W_O[:,h]^T W_O[:,h], block-diagonal.

        What reaches the residual stream is not V but (P V) W_O, so the metric
        on a V-reconstruction error is W_O^T W_O, not the identity. The existing
        `v_scale` is the rank-1 degenerate case of exactly this -- a scalar where
        the right object is a matrix. Measured on this checkpoint: M_V has
        effective rank 116-198 of 256 and condition number 25-267, so the scalar
        is a poor stand-in.

        Free: no calibration, it is built from o_proj's weights alone.

        NOT modelled: attn_output_gate multiplies the attention output before
        o_proj, so the exact metric carries a diag(sqrt(E[gate^2])) factor that
        would need calibration. Omitted -- this is the data-free half.
        """
        Wo = o_proj.weight if hasattr(o_proj, "weight") else o_proj.base.weight
        Wo = Wo.detach().float()                      # (d_model, n_heads*hd)
        n_heads = Wo.shape[1] // head_dim
        per_group = n_heads // n_kv
        blocks = []
        for g in range(n_kv):
            acc = torch.zeros(head_dim, head_dim, device=Wo.device, dtype=Wo.dtype)
            for h in range(g * per_group, (g + 1) * per_group):
                Wh = Wo[:, h * head_dim:(h + 1) * head_dim]
                acc += Wh.T @ Wh
            blocks.append(acc)
        return torch.block_diag(*blocks)              # (n_kv*hd, n_kv*hd)

    def __init__(self, k_proj, v_proj, d_c, balance=True, blend=False,
                 cov=None, v_metric=None):
        super().__init__()
        Wk, k_ref = merged_weight(k_proj)        # (n_kv*hd, d_model)
        Wv, v_ref = merged_weight(v_proj)
        self.k_out, self.v_out = Wk.shape[0], Wv.shape[0]
        d_model = Wk.shape[1]
        self.d_c = int(d_c)

        # --- balance K and V so the joint SVD does not favour the larger ---
        if balance:
            sk = Wk.norm() / math.sqrt(Wk.numel())
            sv = Wv.norm() / math.sqrt(Wv.numel())
            g = (sk * sv).sqrt()
            self.k_scale, self.v_scale = (g / sk).item(), (g / sv).item()
        else:
            self.k_scale = self.v_scale = 1.0

        # Output-side metric on the V block. Two-sided weighted low-rank with a
        # SEPARABLE (Kronecker) weight is still exactly solvable by one SVD --
        # Manton, Mahony & Hua 2003, Thm 3; Markovsky 2019, Thm 4.12 -- so this
        # costs nothing beyond the sqrt/inverse-sqrt of a 512x512. (Sums of two
        # or more Kronecker terms have NO closed form; that boundary is why the
        # K block keeps its scalar here.)
        self.v_mat = v_metric is not None
        if self.v_mat:
            Mv = v_metric.to(Wv.device, torch.float64)
            Mv = 0.5 * (Mv + Mv.T)
            w, Q = torch.linalg.eigh(Mv)
            w = w.clamp_min(w.max() * 1e-8)           # PSD, possibly rank-deficient
            self._mv_half = (Q * w.sqrt()) @ Q.T
            self._mv_ihalf = (Q * w.rsqrt()) @ Q.T
            # trace-normalize so the K/V balance point is unchanged and this
            # measures the SHAPE of the metric, not a rescaling of the V block
            s = (Wv.shape[0] / w.sum()).sqrt()
            self._mv_half = (self._mv_half * s).to(Wv.dtype)
            self._mv_ihalf = (self._mv_ihalf / s).to(Wv.dtype)
            Wv_w = self._mv_half @ (Wv * self.v_scale)
        else:
            Wv_w = Wv * self.v_scale
        W = torch.cat([Wk * self.k_scale, Wv_w], dim=0)

        if cov is not None:
            # --- CARE / SVD-LLM: factorize in the WHITENED basis ---
            L = whiten_factor(cov.to(W.dtype).to(W.device))   # cov = L L^T
            M = W @ L                                          # W S^T
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            r = min(self.d_c, S.numel())
            Linv = torch.linalg.solve_triangular(
                L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype),
                upper=False)
            up = U[:, :r] * S[:r].unsqueeze(0)                 # (out, r)
            down = Vh[:r] @ Linv                               # (r, d_model)
            self.whitened = True
        else:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(self.d_c, S.numel())
            sq = S[:r].sqrt()
            down = (sq.unsqueeze(1) * Vh[:r])                 # (r, d_model)
            up = U[:, :r] * sq.unsqueeze(0)                   # (out, r)
            self.whitened = False

        dev, dt = k_ref.weight.device, k_ref.weight.dtype
        self.down = nn.Linear(d_model, r, bias=False).to(dev, dt)
        self.up_k = nn.Linear(r, self.k_out, bias=False).to(dev, dt)
        self.up_v = nn.Linear(r, self.v_out, bias=False).to(dev, dt)
        with torch.no_grad():
            self.down.weight.copy_(down.to(dt))
            # undo the balancing in the up-projections so the product is the
            # original mapping, not a rescaled one
            self.up_k.weight.copy_((up[:self.k_out] / self.k_scale).to(dt))
            up_v = up[self.k_out:]
            if self.v_mat:                 # un-whiten the output side too
                up_v = self._mv_ihalf @ up_v
            self.up_v.weight.copy_((up_v / self.v_scale).to(dt))

        self.energy_kept = (S[:r].pow(2).sum() / S.pow(2).sum()).item()
        self.full_rank = int(S.numel())

        # optional zero-init blend against the ORIGINAL projections, so the
        # model starts exactly where it was and slides toward the compressed
        # path as `s` learns. Costs the original weights in memory until merged.
        self.blend = blend
        if blend:
            self.k_ref = k_proj
            self.v_ref = v_proj
            for p in (*self.k_ref.parameters(), *self.v_ref.parameters()):
                p.requires_grad_(False)
            self.s = nn.Parameter(torch.tensor(-6.0, device=dev, dtype=torch.float32))

    def forward(self, x):
        c = self.down(x)
        k, v = self.up_k(c), self.up_v(c)
        if self.blend:
            a = torch.sigmoid(self.s).to(k.dtype)
            k = self.k_ref(x) + a * (k - self.k_ref(x))
            v = self.v_ref(x) + a * (v - self.v_ref(x))
        return k, v


class FactoredKProj(nn.Module):
    """Shim so the stock attention forward, which calls k_proj(x) and v_proj(x)
    separately, drives one shared latent. The first call computes c and caches
    both outputs; the second reads the cached V. Keyed on the input tensor's
    identity so it cannot serve a stale value across steps."""

    def __init__(self, latent, which):
        super().__init__()
        self.latent, self.which = latent, which

    def forward(self, x):
        lat = self.latent
        if getattr(lat, "_cached_for", None) is not x:
            lat._cached_k, lat._cached_v = lat(x)
            lat._cached_for = x
        return lat._cached_k if self.which == "k" else lat._cached_v


def whiten_factor(cov, eps=1e-6):
    """S with S^T S = cov, via Cholesky (jittered if near-singular).

    CARE / SVD-LLM: minimizing ||X W^T - X W_hat^T||_F is equivalent to
    minimizing ||(W - W_hat) S^T||_F where S^T S = X^T X. So SVD the WHITENED
    operator W @ S^T, truncate there, and un-whiten -- which optimizes ACTIVATION
    error instead of weight error. Plain SVD optimizes the wrong objective.
    """
    d = cov.shape[0]
    jitter = eps * torch.diag(cov).mean().clamp_min(1e-12)
    for _ in range(6):
        try:
            L = torch.linalg.cholesky(cov + jitter * torch.eye(d, device=cov.device,
                                                               dtype=cov.dtype))
            return L                      # cov = L L^T, so S^T = L
        except Exception:
            jitter *= 10
    # fall back to a symmetric square root
    w, V = torch.linalg.eigh(cov)
    return V @ torch.diag(w.clamp_min(0).sqrt()) @ V.T


def allocate_ranks(spectra, total_budget, min_r=64):
    """CARE's greedy water-filling: start every layer at min_r, then repeatedly
    give one more rank to whichever layer has the highest residual-energy
    priority, until the budget is spent. Puts capacity where the spectrum is
    genuinely complex instead of splitting it evenly."""
    ranks = {k: min_r for k in spectra}
    used = sum(ranks.values())
    import heapq
    def prio(k, r):
        """Marginal squared energy recovered by giving layer k one more rank.

        The objective is to minimise TOTAL truncation error across layers,
        sum_l sum_{j>r_l} S_l[j]^2. Greedily, the next rank should go to whichever
        layer has the largest next singular value squared -- that is the standard
        water-filling solution.

        This was previously S2[r] / tail, which is a RATIO and rewards exactly the
        wrong layers: a fast-decaying spectrum has a small next value but an even
        smaller tail, so the ratio is large and the layer that least needs rank
        gets it. Measured on this model at a 1536 budget, that version handed 1023
        of 1536 to layer 23 (retained energy 0.9917, the best-compressing layer)
        and pinned layers 3, 7 and 15 at the min_r floor despite their being the
        worst (0.9217, 0.9571, 0.9370). The function was written during Stage D
        and never called, so the allocation was never checked against a spectrum.
        """
        S2 = spectra[k].pow(2)
        if r >= S2.numel() - 1:
            return -1.0
        return float(S2[r])
    heap = [(-prio(k, ranks[k]), k) for k in spectra]
    heapq.heapify(heap)
    while used < total_budget and heap:
        negp, k = heapq.heappop(heap)
        if -negp <= 0:
            continue
        ranks[k] += 1
        used += 1
        heapq.heappush(heap, (-prio(k, ranks[k]), k))
    return ranks


def choose_rank(k_proj, v_proj, energy=0.95, min_r=64, max_r=None, balance=True):
    """Smallest rank retaining `energy` of the joint spectrum (YouZhi-style
    per-layer sizing, rather than one global d_c for every layer)."""
    Wk, _ = merged_weight(k_proj)
    Wv, _ = merged_weight(v_proj)
    if balance:
        sk = Wk.norm() / math.sqrt(Wk.numel())
        sv = Wv.norm() / math.sqrt(Wv.numel())
        g = (sk * sv).sqrt()
        Wk, Wv = Wk * (g / sk), Wv * (g / sv)
    S = torch.linalg.svdvals(torch.cat([Wk, Wv], dim=0))
    cum = S.pow(2).cumsum(0) / S.pow(2).sum()
    r = int((cum < energy).sum().item()) + 1
    r = max(min_r, r)
    return min(r, max_r or S.numel()), S


def whitened_spectra(model, covs):
    """Singular values of the operator CARE actually truncates, W @ L.

    Rank allocation must use THESE, not svdvals of the raw weights. Truncating
    the whitened operator at r leaves activation error exactly
    sum_{j>r} sigma_j(WL)^2, because
        E_x||xW^T - xW_hat^T||^2 = ||WL - W_hat L||_F^2      (C = L L^T)
    so squared singular values are activation error in the whitened basis.
    Allocating on raw weight spectra instead optimises ||W - W_hat||, which is
    precisely the objective CARE exists to replace -- measured on this model,
    that mis-specification turned a 5.02% activation-error reduction into 2.94%
    of the wrong quantity.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    out = {}
    for i, l in enumerate(get_trunk(model).layers):
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        Wk, _ = merged_weight(sa.k_proj)
        Wv, _ = merged_weight(sa.v_proj)
        sk = Wk.norm() / math.sqrt(Wk.numel())
        sv = Wv.norm() / math.sqrt(Wv.numel())
        g = (sk * sv).sqrt()
        W = torch.cat([Wk * (g / sk), Wv * (g / sv)], dim=0)
        L = whiten_factor(covs[i].to(W.dtype).to(W.device))
        out[i] = torch.linalg.svdvals(W @ L)
    return out


def convert_to_mla(model, d_c=None, energy=0.95, blend=False, verbose=True,
                   covs=None, budget=None, v_metric=False):
    """Replace k_proj/v_proj on every full-attention layer with a shared latent.

    d_c=None selects per-layer ranks by spectral energy; an int forces one rank
    everywhere (useful as an ablation against the adaptive choice).
    """
    # Argument validation first, before touching the model, so a bad call fails
    # on the reason rather than on an unrelated AttributeError downstream.
    if budget is not None and covs is None:
        raise ValueError(
            "budget allocation requires covs. Allocating on raw weight spectra "
            "optimises ||W - W_hat||, the objective CARE exists to replace.")
    from mercurius.surgery.norm_fusion import get_trunk
    layers = get_trunk(model).layers
    alloc = None
    if budget is not None:
        alloc = allocate_ranks(whitened_spectra(model, covs), int(budget))
        if verbose:
            print(f"  rank allocation over a fixed budget of {int(budget)} "
                  f"(KV ratio unchanged): {dict(sorted(alloc.items()))}")
    info = []
    for i, l in enumerate(layers):
        sa = getattr(l, "self_attn", None)
        if sa is None:
            continue
        if alloc is not None:
            r = alloc[i]
            _, S = choose_rank(sa.k_proj, sa.v_proj, energy=energy)
        elif d_c is None:
            r, S = choose_rank(sa.k_proj, sa.v_proj, energy=energy)
        else:
            r = int(d_c)
            _, S = choose_rank(sa.k_proj, sa.v_proj, energy=energy)
        vm = None
        if v_metric:
            cfg = getattr(model.config, "text_config", model.config)
            vm = LatentKV.v_output_metric(
                sa.o_proj, cfg.num_key_value_heads, cfg.head_dim)
        lat = LatentKV(sa.k_proj, sa.v_proj, r, blend=blend,
                       cov=(covs or {}).get(i), v_metric=vm)
        sa.k_proj = FactoredKProj(lat, "k")
        sa.v_proj = FactoredKProj(lat, "v")
        # deliberately NOT registered as sa._kv_latent: it is already a child
        # of both k_proj and v_proj, and a third path would have it visited
        # three times by any named_modules walk.
        info.append((i, r, lat.full_rank, lat.energy_kept))

    if verbose and info:
        print(f"  converted {len(info)} attention layers to latent KV"
              f"{' (zero-init blend)' if blend else ''}")
        print(f"  {'layer':<8}{'d_c':>6}{'full':>7}{'energy':>9}{'KV saving':>12}")
        for i, r, fr, e in info:
            print(f"  {i:<8}{r:>6}{fr:>7}{e:>9.4f}{fr/r:>11.2f}x")
        tot_before = sum(fr for _, _, fr, _ in info)
        tot_after = sum(r for _, r, _, _ in info)
        print(f"  overall KV cache: {tot_before/tot_after:.2f}x smaller")
    return info
