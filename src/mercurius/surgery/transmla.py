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
    (2 * n_heads * head_dim, d_model), laid out per head as [q_h | gate_h],
    so half its rows are the output gate; naive head-merging corrupts it. Compressing only K and V
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

    VeRA IS HANDLED TOO, and was not until 2026-09-26. VeRALinear also has
    `.base`, so it took the LoRA branch, found no `lora_A`, and returned the
    frozen base -- silently discarding every VeRA delta. That did not matter while
    k_proj/v_proj were unadapted (VERA_ALL lists q_proj and o_proj only), but it
    defeats any staged pipeline that adapts the heads BEFORE compressing them: the
    adaptation would be dropped at exactly the moment it was supposed to inform
    the factorisation.
    """
    if hasattr(mod, "base"):                       # LoRALinear or VeRALinear
        W = mod.base.weight.data.float()
        if getattr(mod, "lora_A", None) is not None:
            W = W + mod.scale * (mod.lora_B.data.float() @ mod.lora_A.data.float())
        elif getattr(mod, "vera_d", None) is not None:
            # dW = diag(b) B diag(d) A, with the two diagonals folded into the
            # shared frozen factors exactly as VeRALinear.forward does.
            I, O = mod.base.in_features, mod.base.out_features
            A = mod.vera_A[:, :I].float() * mod.vera_d.data.float().unsqueeze(1)
            B = mod.vera_B[:O, :].float() * mod.vera_b.data.float().unsqueeze(1)
            W = W + B @ A
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
                 cov=None, v_metric=None, groups=None, head_dim=None,
                 clusters=None):
        """groups: None for one joint latent over all KV heads, or a list of
        (kv_head_indices, rank) -- see _init_grouped.

        clusters: None, or {"covs": (E, d, d), "centroids": (E, d)} from
        experiments/collect_cluster_covs.py, which switches on MIXTURE OF LATENTS:
        E routed (down, up_k, up_v) triples instead of one, each the whitened SVD
        of the SAME W under that cluster's covariance.

        THE ENCODER IS ROUTED TOO, not just the decoder. #35 measured why: with a
        shared `down` the latent carries only the top-r RIGHT-singular coordinates,
        so the discarded energy is gone before any decoder sees it and the
        least-squares-optimal up_k is UNIQUE -- diverse decoders alone provably
        cannot help (+8.5% error at spread 0.3). Routing `down` as well lets each
        expert be the best rank-r fit for its own region, and costs nothing in
        cache: a token only ever needs ITS OWN expert's coordinates, so the cache
        stays r + one index."""
        super().__init__()
        Wk, k_ref = merged_weight(k_proj)        # (n_kv*hd, d_model)
        Wv, v_ref = merged_weight(v_proj)
        # The TRUE projections, kept on CPU (~10 MB/layer, not a buffer, not in the
        # state_dict). mol_cluster_init() fits per-region decoders to K = X W_K^T,
        # which the latent alone cannot supply -- the discarded energy is gone once
        # c = down(x) is computed (#35).
        self._W_ref = (Wk.detach().to("cpu", torch.bfloat16),
                       Wv.detach().to("cpu", torch.bfloat16))
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

        self.groups = None
        self.n_experts = 1 if clusters is None else int(clusters["covs"].shape[0])
        if clusters is not None:
            if groups is not None:
                raise ValueError("clusters and groups are mutually exclusive: MoL "
                                 "wants ONE latent per layer (#38.1 measured "
                                 "grouping as a null anyway)")
            self._init_mol(W, clusters, k_ref, d_model)
            self._finish_init(k_proj, v_proj, blend)
            return
        if groups is not None:
            if self.v_mat:
                raise ValueError("grouped latents do not implement v_metric")
            if cov is None:
                raise ValueError("grouped latents need CARE covariances")
            self._init_grouped(Wk, Wv, cov, groups, head_dim, k_ref, d_model)
            self._finish_init(k_proj, v_proj, blend)
            return

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
        self._finish_init(k_proj, v_proj, blend)

    @torch.no_grad()
    def _init_mol(self, W, clusters, k_ref, d_model):
        """E per-cluster whitened factorisations of the same W. Init only (#42).

        Each expert is built exactly as the single-latent whitened path builds its
        one factorisation -- whiten_factor, SVD, keep top-r -- with that cluster's
        covariance in place of the global one. Nothing else differs, which is why
        this reuses the same algebra rather than a new derivation.

        MEASURED at init (#43), key reconstruction vs the true W_K x, real weights,
        real covariances, real k-means routing:
            plain r=512   0.1633   E=4 r=512  0.0958 (-41%)
            plain r=256   0.3062   E=8 r=256  0.1541  ~= plain r=512, HALF the cache
        Those are INIT numbers and may wash out (#25 saw an allocation advantage
        recover 92%); the durable argument is symmetry breaking (#42.2), since with
        identical experts the pairwise cosine between expert gradients is 0.889 and
        the mechanism may never differentiate in a short run.
        """
        covs, cents = clusters["covs"], clusters["centroids"]
        E = int(covs.shape[0])
        dev, dt = k_ref.weight.device, k_ref.weight.dtype
        downs, upks, upvs, kept = [], [], [], []
        for e in range(E):
            L = whiten_factor(covs[e].to(W.dtype).to(W.device))
            U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
            r = min(self.d_c, S.numel())
            Linv = torch.linalg.solve_triangular(
                L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype),
                upper=False)
            up = U[:, :r] * S[:r].unsqueeze(0)
            downs.append((Vh[:r] @ Linv).to(dt))
            upks.append((up[:self.k_out] / self.k_scale).to(dt))
            up_v = up[self.k_out:]
            if self.v_mat:
                up_v = self._mv_ihalf @ up_v
            upvs.append((up_v / self.v_scale).to(dt))
            kept.append((S[:r].pow(2).sum() / S.pow(2).sum()).item())
        self.r = downs[0].shape[0]
        self.whitened = True
        self.down_w = nn.Parameter(torch.stack(downs).to(dev))
        self.up_k_w = nn.Parameter(torch.stack(upks).to(dev))
        self.up_v_w = nn.Parameter(torch.stack(upvs).to(dev))
        # Router initialised FROM THE CENTROIDS, so it starts matched to the
        # experts. A random router would hand every expert a random 1/E of tokens,
        # which #41.1 measured as leaving the expert gradients 0.889-correlated.
        self.router = nn.Linear(d_model, E, bias=True).to(dev, torch.float32)
        C = cents.to(torch.float32)
        w = C                                            # argmax_e (c_e . x - |c_e|^2/2)
        b = -0.5 * (C * C).sum(1)                        #   == nearest centroid
        # TEMPERATURE. Without it the router gradient is EXACTLY ZERO: centroid
        # norms are O(sqrt(d)) so the logit gaps are O(100), the softmax saturates
        # at a one-hot, and dg/dlogits vanishes -- the router would be frozen on
        # the k-means partition forever, which is the opposite of the learned
        # router this is supposed to be. Dividing every logit by a positive
        # constant leaves the ARGMAX untouched, so the init routing is unchanged
        # while the gradient comes back. tau is taken from the spread of the
        # logits evaluated at the centroids themselves, which needs no data.
        lg = C @ C.t() + b.unsqueeze(0)
        tau = lg.std().clamp_min(1e-6)
        self.router.weight.copy_(w / tau)
        self.router.bias.copy_(b / tau)
        self.router_tau = float(tau)
        self.energy_kept = sum(kept) / len(kept)
        self.full_rank = int(min(W.shape))
        self._mol_last_idx = None

    def _mol_forward(self, x):
        """ROUTED MIXTURE OF LATENTS. Encoder router on x picks e_j; c_j = D_e x_j.

        TIED (#55, the configuration in use): K_j, V_j = U_e c_j with the SAME e_j, so
        the cache holds c_j plus the index e_j (ceil(log2 E) bits). Decoding reads only
        (c_j, e_j) -- verified: the gate factor g/g.detach() is exactly 1.0 in value.
        UNTIED (#54, measured +195% held-out error, kept for reference): a decoder
        router reads c_j alone, d_j = dec_router(c_j), K_j = U_d c_j -- no index, but
        it cannot tell which encoder wrote c_j.
        The dense-mixture branches (ei is None) exist only while annealing and are NOT
        cacheable (they need every expert's latent); inference uses the hard branch.
        With E = 1 either form is exactly plain MLA.
        """
        xf = x.reshape(-1, x.shape[-1])
        ei, ew = self.mol_enc_router(x)
        if ei is None:                                    # annealing: dense mixture
            ew = ew.reshape(-1, self.E_enc).to(x.dtype)
            c = sum(ew[:, e:e + 1] * (xf @ self.down_w[e].t().to(x.dtype))
                    for e in range(self.E_enc))
        else:
            fi = ei.reshape(-1)
            c = xf.new_zeros(xf.shape[0], self.down_w.shape[1])
            for e in range(self.E_enc):
                sel = fi == e
                if sel.any():
                    c[sel] = (xf[sel] @ self.down_w[e].t().to(x.dtype)).to(c.dtype)
            c = c * ew.reshape(-1, 1).to(c.dtype)
        if getattr(self, "mol_tied", False):
            # TIED PAIRS (#55, option 1): decoder e decodes encoder e's latent. At
            # autoregressive decode the index e_j is cached beside c_j -- ceil(log2 E)
            # bits per token per MLA layer, 0.05-0.07% of the latent at 87.5%
            # compression. The encoder ratio is already folded into c above.
            if ei is None:                                 # annealing: dense mixture
                k = sum(ew[:, e:e + 1] * ((xf @ self.down_w[e].t().to(x.dtype))
                                          @ self.up_k_w[e].t().to(x.dtype))
                        for e in range(self.E_enc))
                v = sum(ew[:, e:e + 1] * ((xf @ self.down_w[e].t().to(x.dtype))
                                          @ self.up_v_w[e].t().to(x.dtype))
                        for e in range(self.E_enc))
            else:
                k = c.new_zeros(c.shape[0], self.k_out)
                v = c.new_zeros(c.shape[0], self.v_out)
                for e in range(self.E_enc):
                    sel = fi == e
                    if sel.any():
                        k[sel] = (c[sel] @ self.up_k_w[e].t().to(c.dtype)).to(k.dtype)
                        v[sel] = (c[sel] @ self.up_v_w[e].t().to(c.dtype)).to(v.dtype)
            self._mol_last = (ei, ei)
            return (k.reshape(*x.shape[:-1], self.k_out),
                    v.reshape(*x.shape[:-1], self.v_out))
        di, dw = self.mol_dec_router(c)
        if di is None:
            dw = dw.reshape(-1, self.E_dec).to(c.dtype)
            k = sum(dw[:, d:d + 1] * (c @ self.up_k_w[d].t().to(c.dtype))
                    for d in range(self.E_dec))
            v = sum(dw[:, d:d + 1] * (c @ self.up_v_w[d].t().to(c.dtype))
                    for d in range(self.E_dec))
        else:
            fd = di.reshape(-1)
            k = c.new_zeros(c.shape[0], self.k_out)
            v = c.new_zeros(c.shape[0], self.v_out)
            for d in range(self.E_dec):
                sel = fd == d
                if sel.any():
                    k[sel] = (c[sel] @ self.up_k_w[d].t().to(c.dtype)).to(k.dtype)
                    v[sel] = (c[sel] @ self.up_v_w[d].t().to(c.dtype)).to(v.dtype)
            wr = dw.reshape(-1, 1).to(k.dtype)
            k, v = k * wr, v * wr
        self._mol_last = (ei, di)
        return (k.reshape(*x.shape[:-1], self.k_out),
                v.reshape(*x.shape[:-1], self.v_out))


    def mol_joint_W(self):
        """Rebuild the joint balanced/metric-weighted W exactly as __init__ did."""
        Wk, Wv = (w.float() for w in self._W_ref)
        if self.v_mat:
            Wv_w = self._mv_half.float().cpu() @ (Wv * self.v_scale)
        else:
            Wv_w = Wv * self.v_scale
        return torch.cat([Wk * self.k_scale, Wv_w], dim=0)

    @torch.no_grad()
    def mol_factor(self, cov, r, G=None, damp=1e-3):
        """ONE whitened-SVD factorization under `cov`: identical math to the
        ungrouped constructor branch, so at E=1 with the CARE covariance it
        reproduces our MLA init. Returns (down (r,d), up_k (k_out,r), up_v).

        G: optional OUTPUT FISHER over the joint [k; v] outputs in NATURAL units
        (capture_kv_fisher). Then the fit minimises E||G^1/2 (W - W^) x||^2 -- the
        second-order loss proxy -- exactly, since input cov (x) output Fisher is a
        Kronecker-separable weight (Manton, Mahony & Hua 2003): SVD of G^1/2 W L,
        up = G^-1/2 U S. The K/V balance scales are then meaningless and unused.
        Measured offline (#56): held-out Fisher error at equal rank ~ a 32-35% cache
        saving over the CARE fit, per layer x0.36-x0.85 on the dCE proxy."""
        if G is not None:
            dev = cov.device
            Wk, Wv = (w.to(dev, torch.float32) for w in self._W_ref)
            W = torch.cat([Wk, Wv], 0)
            ev, Q = torch.linalg.eigh(G.to("cpu", torch.float64))
            ev = ev.clamp_min(0) + damp * float(ev.clamp_min(0).mean())
            A = ((Q * ev.sqrt()) @ Q.T).to(dev, torch.float32)
            Ai = ((Q * ev.rsqrt()) @ Q.T).to(dev, torch.float32)
            # fp32 here: 8x faster than fp64 on this GPU, agreement ~1e-4 (#56)
            L = whiten_factor(cov.to(torch.float32))
            U, S, Vh = torch.linalg.svd(A @ W @ L, full_matrices=False)
            Linv = torch.linalg.solve_triangular(
                L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
            up = Ai @ (U[:, :r] * S[:r].unsqueeze(0))
            return Vh[:r] @ Linv, up[:self.k_out], up[self.k_out:]
        W = self.mol_joint_W().to(cov.device, torch.float64)
        L = whiten_factor(cov.to(torch.float64))
        U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
        Linv = torch.linalg.solve_triangular(
            L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
        up = U[:, :r] * S[:r].unsqueeze(0)
        down = Vh[:r] @ Linv
        up_k = up[:self.k_out] / self.k_scale
        up_v = up[self.k_out:]
        if self.v_mat:
            up_v = self._mv_ihalf.to(up_v.device, up_v.dtype) @ up_v
        return down, up_k, up_v / self.v_scale

    @torch.no_grad()
    def _init_grouped(self, Wk, Wv, cov, groups, head_dim, k_ref, d_model):
        """One latent per GROUP of KV heads, laid out as a single LatentKV.

        Each group g (a set of KV heads) gets its own CARE-whitened SVD of just
        its K and V rows, truncated at rank r_g. The layer's down projection is
        the concatenation of the group downs; up_k / up_v are block-structured,
        head h's rows reading only its group's columns and zeros elsewhere.
        That is exactly separate per-group latents (cache = sum r_g), expressed
        as one module, so training, the rebuild and the absorbed-MLA path need
        no change. The zero blocks are ordinary parameters: training may fill
        them, letting a head read another group's latent at no cache cost -- the
        grouping sets the initial factorisation, not a constraint.

        Why group at all. A joint SVD over every KV head spends rank on
        whichever heads carry the most whitened energy, which is not the same
        as the heads that retrieve (findings 0.7: the spectral allocator gave
        the most rank to the layer that retrieves least). Grouping by retrieval
        score lets retrieval heads be factorised among themselves and given
        more rank, at the same total cache.
        """
        hd = int(head_dim)
        L = whiten_factor(cov.to(torch.float32).to(Wk.device))
        Linv = torch.linalg.solve_triangular(
            L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
        R = sum(int(r) for _, r in groups)
        up_k = torch.zeros(self.k_out, R, device=Wk.device)
        up_v = torch.zeros(self.v_out, R, device=Wk.device)
        downs, kept, total, col = [], 0.0, 0.0, 0
        seen = sorted(h for heads, _ in groups for h in heads)
        if seen != list(range(self.k_out // hd)):
            raise ValueError(f"groups must partition the KV heads exactly once, got {groups}")
        for heads, r in groups:
            rows = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in heads]).to(Wk.device)
            Wg = torch.cat([Wk[rows] * self.k_scale, Wv[rows] * self.v_scale], 0)
            U, S, Vh = torch.linalg.svd(Wg @ L, full_matrices=False)
            r = min(int(r), S.numel())
            up = U[:, :r] * S[:r].unsqueeze(0)
            downs.append(Vh[:r] @ Linv)
            nk = rows.numel()
            up_k[rows, col:col + r] = up[:nk] / self.k_scale
            up_v[rows, col:col + r] = up[nk:] / self.v_scale
            kept += float(S[:r].pow(2).sum()); total += float(S.pow(2).sum())
            col += r
        R = col
        dev, dt = k_ref.weight.device, k_ref.weight.dtype
        self.down = nn.Linear(d_model, R, bias=False).to(dev, dt)
        self.up_k = nn.Linear(R, self.k_out, bias=False).to(dev, dt)
        self.up_v = nn.Linear(R, self.v_out, bias=False).to(dev, dt)
        self.down.weight.copy_(torch.cat(downs, 0).to(dt))
        self.up_k.weight.copy_(up_k[:, :R].to(dt))
        self.up_v.weight.copy_(up_v[:, :R].to(dt))
        self.d_c = R
        self.whitened = True
        self.groups = [(list(h), int(r)) for h, r in groups]
        self.energy_kept = kept / max(total, 1e-30)
        self.full_rank = 2 * self.k_out

    def _finish_init(self, k_proj, v_proj, blend):
        # optional zero-init blend against the ORIGINAL projections, so the
        # model starts exactly where it was and slides toward the compressed
        # path as `s` learns. Costs the original weights in memory until merged.
        self.blend = blend
        if blend:
            self.k_ref = k_proj
            self.v_ref = v_proj
            for p in (*self.k_ref.parameters(), *self.v_ref.parameters()):
                p.requires_grad_(False)
            self.s = nn.Parameter(torch.tensor(-6.0, device=self.down.weight.device,
                                               dtype=torch.float32))


    def forward(self, x):
        if getattr(self, "mol_struct", None):              # #57 arm C (mol_struct.py)
            from mercurius.surgery.mol_struct import mol_struct_forward
            k, v = mol_struct_forward(self, x)
            if not self.blend:
                return k, v
        elif getattr(self, "mol_routed", False):
            k, v = self._mol_forward(x)
            if not self.blend:
                return k, v
        else:
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
                   covs=None, budget=None, v_metric=False, alloc=None,
                   groups=None, only=None):
    """groups: optional {trunk_layer: [(kv_head_indices, rank), ...]} -- one
    latent per group of KV heads (LatentKV._init_grouped); a layer's d_c is
    then the sum of its group ranks. Overrides d_c / alloc / budget there.

    only: optional set of trunk-layer indices to convert, leaving the others
    untouched; layers already converted are always skipped. This is what makes
    SEQUENTIAL calibration possible (calibrate_mla_sequential): convert layer l,
    then re-collect the next layer's inputs from the partly compressed model."""
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
    if alloc is not None and budget is not None:
        raise ValueError("pass alloc or budget, not both: budget recomputes an "
                         "allocation and would discard the explicit one")
    if alloc is not None:
        alloc = {int(k): int(v) for k, v in alloc.items()}
        if verbose:
            print(f"  explicit rank allocation, total {sum(alloc.values())}: "
                  f"{dict(sorted(alloc.items()))}", flush=True)
    if budget is not None:
        alloc = allocate_ranks(whitened_spectra(model, covs), int(budget))
        if verbose:
            print(f"  rank allocation over a fixed budget of {int(budget)} "
                  f"(KV ratio unchanged): {dict(sorted(alloc.items()))}")
    info = []
    for i, l in enumerate(layers):
        sa = getattr(l, "self_attn", None)
        if sa is None or isinstance(sa.k_proj, FactoredKProj):
            continue
        if only is not None and i not in only:
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
        g_i = (groups or {}).get(i)
        if g_i is not None:
            cfg = getattr(model.config, "text_config", model.config)
            r = sum(int(rr) for _, rr in g_i)
        lat = LatentKV(sa.k_proj, sa.v_proj, r, blend=blend,
                       cov=(covs or {}).get(i), v_metric=vm, groups=g_i,
                       head_dim=(cfg.head_dim if g_i is not None else None))
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
