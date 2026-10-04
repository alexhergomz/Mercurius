"""Mixture of Latents -- a per-token switch over the MLA latent DECODERS.

    c_j = down(x_j)                      cached, r-dimensional, SHARED encoder
    e_j = argmax router(x_j)             cached, one small index per token
    K_j = up_k^(e_j) c_j                 E alternative decoders
    V_j = up_v^(e_j) c_j

WHY THIS ESCAPES THE RANK CEILING, which latent_ext.py's docstring says nothing
else does. That argument -- "every key lies in range(up_k), an r-dimensional
subspace, so the score is a bilinear form of rank <= r" -- assumes ONE shared
up_k. Here each key still lies in an r-dimensional subspace, but DIFFERENT TOKENS
LIE IN DIFFERENT ONES, so the key set collectively spans up to E*r dimensions.
The score stops being a single bilinear form and becomes E of them, selected per
token. Cache cost is r + one index (2 bits at E=4), not E*r.

Contrast the three mechanisms now in the codebase, since they are easy to confuse:
    r -> 2r         one 2r subspace for every key;        cache DOUBLES
    MultiTapUp      K_j = sum_i A_i c_{j-i}, independent A_i, up to (k+1)r;
                    cache unchanged, but the SAME map for every token
    ConvUp          A_b = up diag(w_b), all sharing range(up); NO ceiling change
    MoL (here)      one of E r-dim subspaces, CHOSEN PER TOKEN; cache + 1 index

BRANCH THE DECODER, SHARE THE ENCODER. Branching `down` instead would leave every
key inside a single range(up_k) and buy no ceiling at all, so `down` stays shared:
that also halves the parameter cost and keeps the cached vector semantically
uniform across tokens.

ABSORPTION SURVIVES ON BOTH SIDES, which is what makes this shippable.
  query side   q^T K_j = (up_k^(e_j)T q)^T c_j. Precompute E absorbed queries per
               decode step; each cached key then costs the same r multiply-adds
               as plain MLA. E x per-step work, nothing per key.
  value side   sum_j a_j up_v^(e_j) c_j does NOT factor the way plain MLA's does,
               but it REGROUPS:
                   sum_e up_v^(e) ( sum_{j: e_j=e} a_j c_j )
               i.e. E latent accumulators and E up-projections PER STEP. Again
               E x per-step, nothing per key. Verified numerically in
               check_absorption_equivalence() below.

EXACT AT INIT, and for a reason worth noting: every expert is initialised as a
COPY of the original up-projection, so the output is identical no matter what the
router does. That means the install is function-preserving REGARDLESS of router
initialisation -- the usual MoE cold-start problem does not apply to exactness.
It does still apply to learning: at init the router has no gradient signal that
distinguishes experts, because they all compute the same thing. Experts diverge
only once the router's own noise breaks the tie, which is why the balance loss
and the routing stats below exist.

TOP-1 STAYS DIFFERENTIABLE via the ratio trick: the chosen expert's output is
multiplied by g_e / g_e.detach(), which is exactly 1.0 in the forward pass -- so
exactness at init is preserved -- while d/d(router) still flows through g_e.
Switch Transformer multiplies by g_e itself, which would scale the output by
~1/E at init and break the function-preserving property we rely on here.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoLState:
    """Per-LatentKV holder passed between the router and the decoders.

    The router sees x (d_model) inside the `down` call; the decoders see c (r)
    inside the `up_k`/`up_v` calls. This carries the routing decision between
    them. Deliberately a plain object, not a buffer: it is transient per forward
    pass and must never be checkpointed.
    """

    def __init__(self, n_experts):
        self.E = n_experts
        self.idx = None          # (..., T) long, chosen expert per token
        self.ratio = None        # (..., T, 1) float, g_e / g_e.detach()
        self.aux = None          # scalar, Switch load-balance loss
        self.counts = torch.zeros(n_experts)   # cumulative, for collapse checks


class MoLRouter(nn.Module):
    """Wraps a LatentKV's `down`: routes on x, then applies the SHARED encoder."""

    def __init__(self, down, state, n_experts, balance=0.01, noise=0.0):
        super().__init__()
        self.down, self.state = down, state
        self.E, self.balance, self.noise = n_experts, balance, noise
        self.router = nn.Linear(down.in_features, n_experts, bias=False,
                                device=down.weight.device, dtype=torch.float32)
        # Small random init, NOT zeros: identical experts already guarantee
        # exactness, so the router's only job at init is to break the tie. A zero
        # router gives a uniform argmax (always expert 0) and the other E-1
        # decoders would never receive a single gradient.
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x):
        logits = self.router(x.float())
        if self.noise > 0.0 and self.training:
            logits = logits + self.noise * torch.randn_like(logits)
        p = logits.softmax(-1)
        idx = p.argmax(-1)                                  # hard top-1
        g = p.gather(-1, idx.unsqueeze(-1))                 # (..., 1)
        self.state.idx = idx
        self.state.ratio = g / g.detach()                   # == 1.0 forward
        # Switch Transformer aux balance loss: alpha * E * sum_e f_e * P_e, with
        # f_e the fraction of tokens routed to e and P_e the mean router prob.
        # Without it top-1 routing collapses; with identical-init experts that
        # risk is higher still, since nothing distinguishes them early.
        f = F.one_hot(idx, self.E).float().reshape(-1, self.E).mean(0)
        P = p.reshape(-1, self.E).mean(0)
        self.state.aux = self.balance * self.E * (f * P).sum()
        if not torch.is_grad_enabled():
            self.state.counts = self.state.counts.to(f.device) + f.detach().cpu() \
                if f.device.type == "cpu" else \
                self.state.counts + f.detach().to(self.state.counts.device)
        return self.down(x)


class MoLUp(nn.Module):
    """Wraps `up_k` or `up_v` with E decoders, selected by the shared routing."""

    def __init__(self, up, state, n_experts, spread=0.0):
        super().__init__()
        self.up, self.state, self.E = up, state, n_experts
        r, out = up.in_features, up.out_features
        w = up.weight.detach().clone().float()          # (out, r)
        if spread <= 0.0:
            # E copies of the ORIGINAL weight => exact at init for any routing.
            # BUT the router then has ZERO output-based gradient at step 0, since
            # every expert computes the same thing. MoELoRA (2402.12851) reports
            # exactly this: with under-differentiated experts "the gating network
            # shows no preference for any specific expert, resulting in a routing
            # process that appears random". Router dynamics under identical-copy
            # init are an unstudied gap in the literature.
            self.experts = nn.Parameter(w.unsqueeze(0).repeat(n_experts, 1, 1))
        else:
            # DIVERSE INIT. Each expert gets the base decoder plus a distinct
            # block of directions from the ORTHOGONAL COMPLEMENT of its column
            # space, so span(up^(e)) differs per expert and the union reaches
            # beyond the single r-dimensional subspace MLA truncated to.
            #
            # Exactness is deliberately given up, and that is not a regression:
            # MLA is ALREADY a lossy rank-r truncation, so the arm we compare
            # against is inexact too. A mixture of distinct rank-r decoders whose
            # union covers more of the original map is a BETTER starting point
            # than one global rank-r fit -- the piecewise-linear-approximation
            # argument. What it cannot do is let any single token see the
            # full-rank map: a rank-r decoder applied to c_j cannot reproduce
            # W_K x_j for arbitrary x_j. The union spans more; each token still
            # gets one r-dimensional slice, chosen for it by the router.
            #
            # NOTE this picks complement directions WITHOUT knowing which of the
            # discarded directions actually carry energy. Aligning them to the
            # discarded singular directions of the original W_K, or to per-cluster
            # least-squares fits, needs a calibration pass (see mol_cluster_init).
            # THE UNION IS CAPPED AT min(E*r, out), not E*r. The complement of
            # span(w) has only out - r dimensions, so for the real shapes here
            # (out = kv_heads*head_dim = 1024, r ~ 512) it holds ~512 directions:
            # disjoint per-expert blocks are IMPOSSIBLE beyond E=2. Experts must
            # therefore overlap, and the benefit past that point is not "spans
            # more" but "each token gets a DIFFERENT r-dim slice of the full
            # out-dimensional key space" -- the piecewise-linear gain, which is
            # the part that does not saturate at E=2.
            base_cols = torch.linalg.qr(w)[0][:, :r]              # (out, r)
            proj = torch.eye(out, dtype=torch.float32) - base_cols @ base_cols.t()
            ws = []
            for e in range(n_experts):
                # A DISTINCT random direction set per expert, drawn inside the
                # complement. An earlier version indexed disjoint blocks by
                # (e*r) % (cols - r), which WRAPPED and handed experts 0 and 2
                # the identical subspace (principal-angle cos 1.0) -- only E/2
                # distinct decoders for E experts.
                gen = torch.Generator().manual_seed(1234 + 7919 * e)
                blk = proj @ torch.randn(out, r, generator=gen, dtype=torch.float32)
                blk = torch.linalg.qr(blk)[0][:, :r]
                ws.append(w + spread * blk * w.norm() / max(blk.norm().item(), 1e-6))
            self.experts = nn.Parameter(torch.stack(ws))

    def forward(self, c):
        idx, ratio = self.state.idx, self.state.ratio
        if idx is None:                       # router did not run: stay exact
            return self.up(c)
        cf = c.float()
        flat_c = cf.reshape(-1, cf.shape[-1])
        flat_i = idx.reshape(-1)
        out = flat_c.new_zeros(flat_c.shape[0], self.experts.shape[1])
        # Masked per-expert matmuls. E small gathers beats materialising a
        # per-token (out x r) weight tensor, which would be B*T*out*r floats.
        for e in range(self.E):
            m = flat_i == e
            if m.any():
                out[m] = flat_c[m] @ self.experts[e].t()
        out = out.reshape(*cf.shape[:-1], -1)
        return (out * ratio).to(c.dtype)


def install_mol(model, n_experts=4, balance=0.01, noise=0.0, spread=0.0,
                verbose=True):
    """Install Mixture of Latents on every LatentKV. Exact at init.

    NOTE the wrappers RENAME what they wrap (down -> down.down, up_k -> up_k.up)
    exactly like GatedLatent/MultiTapUp/ConvDown, so a rebuild MUST install this
    before loading a checkpoint or the projections themselves go unmatched and
    strict=False silently drops them. Same failure mode as #23.

    Mutually exclusive with --mla-gate/--mla-taps/--mla-conv: they collide on the
    same wrapped attribute names.
    """
    from mercurius.surgery.transmla import LatentKV
    lats = [m for m in model.modules() if isinstance(m, LatentKV)]
    if not lats:
        raise SystemExit(
            "--mla-mol found no LatentKV: MLA is not installed, so there are no "
            "latent decoders to mix. Pass --mla-dc/--mla-groups.")
    n = 0
    for lat in lats:
        st = MoLState(n_experts)
        lat._mol_state = st
        lat.down = MoLRouter(lat.down, st, n_experts, balance, noise)
        lat.up_k = MoLUp(lat.up_k, st, n_experts, spread)
        lat.up_v = MoLUp(lat.up_v, st, n_experts, spread)
        n += 1
    if verbose:
        print(f"  Mixture of Latents: {n_experts} decoders on {n} layers, shared "
              f"encoder, top-1 routing (balance {balance:g}, noise {noise:g}); "
              f"cache + {math.ceil(math.log2(n_experts))} bits/token; ceiling "
              f"r -> up to {n_experts}r; "
              + ("exact at init" if spread <= 0 else
                 f"DIVERSE init spread={spread:g} (not exact; union of distinct "
                 f"rank-r decoders covers more than one global fit)"), flush=True)
    return n


def mol_aux_loss(model):
    """Sum of the per-layer MoL auxiliary losses (Switch balance for the x-routed and
    legacy routers; the optional variance loss for the cosine router). None if none."""
    tot, n = 0.0, 0
    for m in model.modules():
        if isinstance(m, MoLRouter) and m.state.aux is not None:
            tot = tot + m.state.aux
            n += 1
        elif isinstance(m, LatentRouter) and m.aux is not None:
            tot = tot + m.aux
            n += 1
    return tot if n else None


def mol_routing_report(model):
    """Per-layer expert usage, for spotting collapse. Returns a list of lists."""
    rows = []
    for m in model.modules():
        if isinstance(m, MoLRouter):
            c = m.state.counts
            s = float(c.sum()) or 1.0
            rows.append([round(float(v) / s, 3) for v in c])
    return rows


def check_absorption_equivalence(seed=0, E=4, r=16, d_out=24, T=32, tol=1e-5):
    """The load-bearing claim: the DECODE regrouping equals the naive value path.

    naive     o = sum_j a_j up_v^(e_j) c_j          (what training computes)
    decode    o = sum_e up_v^(e) (sum_{j:e_j=e} a_j c_j)
    If these differ, MoL is not shippable, because decode could not avoid
    materialising V. Exercised as a unit test rather than asserted in prose.
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(E, d_out, r, generator=g)
    c = torch.randn(T, r, generator=g)
    e = torch.randint(0, E, (T,), generator=g)
    a = torch.randn(T, generator=g).softmax(0)

    naive = sum(a[j] * (W[e[j]] @ c[j]) for j in range(T))
    decode = torch.zeros(d_out)
    for k in range(E):
        m = e == k
        if m.any():
            decode = decode + W[k] @ (a[m].unsqueeze(-1) * c[m]).sum(0)
    err = (naive - decode).abs().max().item()
    scale = naive.abs().max().item()
    return err, scale, err <= tol * max(scale, 1.0)


# ------------------------------------------------- latent-routed decoders (#44)
class LatentRouter(nn.Module):
    """Routes on the CACHED LATENT, so the expert index is never stored.

    e_j = argmax router(c_j). c_j is already in the cache, so e_j is recomputable at
    decode and the cache stays BYTE-IDENTICAL to plain MLA.

    TWO MODES.
      "cosine" (DEFAULT, #53): scores are scale * cos(c, w_e), with w_e initialised
          to the L2-normalised k-means centroids -- the Cluster-Aware Upcycling recipe
          (arXiv 2604.13508). Cosine routing BOUNDS the logits, which is why that
          paper needs no temperature and why our raw-dot-product centroid router
          saturated to an EXACTLY ZERO gradient.
          BALANCING is DeepSeek-V3's auxiliary-loss-free bias (arXiv 2412.19437): a
          per-expert bias added to the SELECTION score only, excluded from the gating
          weight, nudged by observed load. No aux loss, no balancing coefficient in
          the objective, no interference gradient.
      "dot" (LEGACY): the #52 router, kept so those checkpoints still replay.

    THE BIAS IS A Parameter, NOT A BUFFER, ON PURPOSE. save_trainable() persists only
    requires_grad parameters, so a buffer would be DROPPED at save and evaluation
    would select experts WITHOUT the bias the model trained with -- silently. The
    bias only enters an argmax, so its .grad stays None and AdamW skips it; it moves
    solely through mol_update_bias(), called once per optimizer step.
    """

    def __init__(self, r, n_experts, centroids=None, device=None, balance=0.0,
                 noise=0.0, anneal=0, mode="cosine", scale=10.0, var_coef=0.0):
        super().__init__()
        self.E, self.mode = n_experts, mode
        self.balance, self.noise, self.var_coef = balance, noise, var_coef
        # ANNEAL is an AMPLIFIER, not a symmetry breaker: arXiv 2605.02124 section 6.1
        # states that at exact symmetry the output is independent of the router.
        self.anneal = int(anneal)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long),
                             persistent=False)
        self.aux = None
        # load of the LAST forward, OVERWRITTEN not accumulated: grad checkpointing
        # recomputes the forward in backward, and an accumulator would double-count.
        self.register_buffer("last_load", torch.zeros(n_experts), persistent=False)
        self.register_buffer("counts", torch.zeros(n_experts), persistent=False)
        if mode == "dot":
            self.lin = nn.Linear(r, n_experts, bias=True, device=device,
                                 dtype=torch.float32)
            nn.init.normal_(self.lin.weight, std=0.02)
            nn.init.zeros_(self.lin.bias)
        elif mode == "cosine":
            w = (torch.randn(n_experts, r, device=device) * 0.02 if centroids is None
                 else centroids.to(device=device, dtype=torch.float32))
            self.w = nn.Parameter(w.float())
            self.log_scale = nn.Parameter(
                torch.tensor(math.log(scale), device=device, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(n_experts, device=device,
                                                 dtype=torch.float32))
        else:
            raise ValueError(f"unknown router mode {mode!r}")

    def scores(self, c):
        if self.mode == "dot":
            return self.lin(c.float())
        cn = F.normalize(c.float(), dim=-1)
        wn = F.normalize(self.w, dim=-1)
        return (cn @ wn.t()) * self.log_scale.exp()

    def forward(self, c):
        logits = self.scores(c)
        if self.noise > 0.0 and self.training:
            logits = logits + self.noise * torch.randn_like(logits)
        p = logits.softmax(-1)
        sel = logits if self.mode == "dot" else logits + self.bias.detach()
        idx = sel.argmax(-1)
        g = p.gather(-1, idx.unsqueeze(-1))           # gating uses the UNBIASED score

        f = F.one_hot(idx, self.E).float().reshape(-1, self.E).mean(0)
        self.last_load = f.detach()
        aux = None
        if self.mode == "dot" and self.balance > 0:
            P = p.reshape(-1, self.E).mean(0)
            aux = self.balance * self.E * (f * P).sum()
        if self.var_coef > 0:
            # VARIANCE LOSS (arXiv 2505.22323): maximise each expert's routing-score
            # variance ACROSS TOKENS, i.e. make routing discriminative. Coefficient is
            # NOT stated in the paper's reachable text -- var_coef is our choice.
            pv = p.reshape(-1, self.E)
            lv = -pv.var(0, unbiased=False).mean()
            aux = self.var_coef * lv if aux is None else aux + self.var_coef * lv
        self.aux = aux
        if not torch.is_grad_enabled():
            self.counts = self.counts.to(f.device) + f.detach()

        if self.training and self.anneal > 0 and int(self._step) < self.anneal:
            t = 1.0 - float(self._step) / self.anneal
            self._step += 1
            return None, (logits / max(t * 4.0, 1e-2)).softmax(-1)
        return idx, g / g.detach()          # ratio is exactly 1.0 forward


@torch.no_grad()
def mol_update_bias(model, gamma=1e-3):
    """DeepSeek-V3 bias step: b_e += gamma * sign(mean_load - load_e). Call ONCE per
    optimizer step. Returns the max load imbalance seen (max/mean), for logging."""
    worst = 0.0
    for m in model.modules():
        if isinstance(m, LatentRouter) and m.mode == "cosine":
            load = m.last_load.to(m.bias.device)
            if float(load.sum()) == 0:
                continue
            m.bias.add_(gamma * torch.sign(load.mean() - load))
            worst = max(worst, float(load.max() / load.mean().clamp_min(1e-9)))
    return worst


def _spherical_kmeans(Z, E, iters=30, restarts=4, seed=0):
    """k-means++ on the unit sphere (cosine), best of `restarts`. Z: (N, r) cuda."""
    Zn = F.normalize(Z.float(), dim=-1)
    g = torch.Generator(device=Zn.device).manual_seed(seed)
    best = None
    for _ in range(restarts):
        C = [Zn[torch.randint(Zn.shape[0], (1,), generator=g, device=Zn.device)][0]]
        d = 1 - Zn @ C[0]
        for _k in range(E - 1):
            pr = d.clamp_min(0)
            pr = pr / pr.sum() if float(pr.sum()) > 0 else torch.ones_like(pr) / len(pr)
            C.append(Zn[int(torch.multinomial(pr, 1, generator=g))])
            d = torch.minimum(d, 1 - Zn @ C[-1])
        C = torch.stack(C)
        for _it in range(iters):
            a = (Zn @ C.t()).argmax(1)
            newC = torch.stack([F.normalize(Zn[a == e].mean(0), dim=0) if (a == e).any()
                                else C[e] for e in range(E)])
            if torch.allclose(newC, C, atol=1e-6):
                break
            C = newC
        a = (Zn @ C.t()).argmax(1)
        score = float((Zn * C[a]).sum())              # total cosine, higher better
        if best is None or score > best[0]:
            best = (score, C, a)
    return best[1], best[2]


@torch.no_grad()
def mol_cluster_init(model, X_by_lat, ridge=1e-3, min_ratio=1.5, verbose=True):
    """Cluster-aware init, adapted to latent-routed decoders (#53).

    Per LatentKV: c = down(X); spherical k-means on c; router w <- normalised
    centroids; for each cluster, each expert's up_k / up_v <- the RIDGED
    LEAST-SQUARES decoder from c to the TRUE K = X W_K^T and V = X W_V^T (the merged
    projections stashed on the LatentKV at construction). A cluster with fewer than
    min_ratio * r rows keeps its global decoder rather than overfit.

    X_by_lat: {LatentKV module: (N, d_model) tensor of inputs collected from the
    FULLY BUILT student}, so the fit is on exactly the distribution the arm trains
    on. Returns a list of per-layer report dicts.
    """
    from mercurius.surgery.transmla import LatentKV
    reports = []
    for lat, X in X_by_lat.items():
        if not isinstance(lat, LatentKV) or not hasattr(lat, "latent_router"):
            continue
        rt = lat.latent_router
        E = rt.E
        dev = rt.w.device if rt.mode == "cosine" else rt.lin.weight.device
        Wk, Wv = (w.to(dev, torch.float32) for w in lat._W_ref)
        Xf = X.to(dev, torch.float32)
        c = lat.down(Xf.to(lat.down.weight.dtype)).float()
        K, V = Xf @ Wk.t(), Xf @ Wv.t()
        r = c.shape[1]

        def fit(cm, T):
            G = cm.t() @ cm
            G += ridge * float(G.diagonal().mean()) * torch.eye(r, device=dev)
            return torch.linalg.solve(G, cm.t() @ T).t()        # (out, r)

        gk, gv = fit(c, K), fit(c, V)
        def err(pk, pv):
            return float(((pk - K).norm() ** 2 + (pv - V).norm() ** 2)
                         / (K.norm() ** 2 + V.norm() ** 2)) ** 0.5
        e_plain = err(lat.up_k.up(c.to(lat.up_k.up.weight.dtype)).float(),
                      lat.up_v.up(c.to(lat.up_v.up.weight.dtype)).float())
        e_global = err(c @ gk.t(), c @ gv.t())

        C, a = _spherical_kmeans(c, E)
        sizes = torch.bincount(a, minlength=E).tolist()
        pk, pv = torch.empty_like(K), torch.empty_like(V)
        kept = 0
        for e in range(E):
            m = a == e
            if int(m.sum()) >= min_ratio * r:
                ek, ev = fit(c[m], K[m]), fit(c[m], V[m])
                kept += 1
            else:
                ek, ev = gk, gv
            lat.up_k.experts[e].copy_(ek)
            lat.up_v.experts[e].copy_(ev)
            if m.any():
                pk[m], pv[m] = c[m] @ ek.t(), c[m] @ ev.t()
        e_clu = err(pk, pv)
        if rt.mode == "cosine":
            rt.w.copy_(C)
            rt.bias.zero_()
        ex = F.normalize(lat.up_k.experts.detach().flatten(1), dim=1)
        cs = ex @ ex.t()
        rep = dict(r=r, sizes=sizes, fitted=kept, err_plain=e_plain,
                   err_global=e_global, err_cluster=e_clu,
                   expert_cos=float(cs[~torch.eye(E, dtype=bool, device=cs.device)].mean()))
        reports.append(rep)
        if verbose:
            print(f"    MoL cluster init r={r:>4}: sizes {sizes}  fitted {kept}/{E}  "
                  f"rel err plain {e_plain:.4f} -> global LS {e_global:.4f} -> "
                  f"per-cluster {e_clu:.4f}  expert cos {rep['expert_cos']:.4f}",
                  flush=True)
    return reports


class LatentRoutedUp(nn.Module):
    """E decoders over a SHARED latent, selected by a shared LatentRouter.

    The router is held in a LIST so nn.Module does not register it twice (it is
    shared between up_k and up_v, which must agree on the expert). The state_dict
    therefore carries exactly one copy, under the LatentKV.
    """

    def __init__(self, up, router, n_experts):
        super().__init__()
        self.up, self.E = up, n_experts
        self._router = [router]                   # hidden from registration
        w = up.weight.detach().clone().float()
        self.experts = nn.Parameter(w.unsqueeze(0).repeat(n_experts, 1, 1))

    def forward(self, c):
        idx, w = self._router[0](c)
        cf = c.float()
        flat = cf.reshape(-1, cf.shape[-1])
        if idx is None:                       # ANNEALING: dense weighted mixture
            wf = w.reshape(-1, self.E)
            out = sum(wf[:, e:e + 1] * (flat @ self.experts[e].t())
                      for e in range(self.E))
            return out.reshape(*cf.shape[:-1], -1).to(c.dtype)
        fi = idx.reshape(-1)
        out = flat.new_zeros(flat.shape[0], self.experts.shape[1])
        for e in range(self.E):
            m = fi == e
            if m.any():
                out[m] = flat[m] @ self.experts[e].t()
        out = out.reshape(*cf.shape[:-1], -1) * w
        return out.to(c.dtype)


def install_mol_latent(model, n_experts=4, centroids=None, balance=0.0,
                       noise=0.0, anneal=0, mode="cosine", scale=10.0, var_coef=0.0,
                       verbose=True):
    """Latent-routed decoders on every LatentKV, E identical copies at install.

    Copies alone do NOT work: #52 measured experts still at pairwise cosine 0.9996
    after 150 steps, because at exact symmetry the router gets no gradient. Follow
    this with mol_cluster_init() for a differentiated start (#53).
    """
    from mercurius.surgery.transmla import LatentKV
    lats = [(n, m) for n, m in model.named_modules() if isinstance(m, LatentKV)]
    if not lats:
        raise SystemExit(
            "--mla-mol-latent found no LatentKV: MLA is not installed, so there is "
            "no latent to route on. Pass --mla-dc/--mla-groups.")
    n = 0
    for name, lat in lats:
        r = lat.up_k.in_features
        dev = lat.up_k.weight.device
        cen = None if centroids is None else centroids.get(n, centroids.get(name))
        router = LatentRouter(r, n_experts, centroids=cen, device=dev,
                              balance=balance, noise=noise, anneal=anneal,
                              mode=mode, scale=scale, var_coef=var_coef)
        lat.latent_router = router                # registered ONCE, here
        lat.up_k = LatentRoutedUp(lat.up_k, router, n_experts)
        lat.up_v = LatentRoutedUp(lat.up_v, router, n_experts)
        n += 1
    if verbose:
        bal = ("DeepSeek selection bias (no aux loss)" if mode == "cosine"
               else f"Switch aux {balance:g}")
        print(f"  MoL (latent-routed decoders): {n_experts} decoders on {n} layers, "
              f"{mode} router, {bal}"
              + (f", variance loss {var_coef:g}" if var_coef else "")
              + "; cache unchanged (no index)", flush=True)
    return n


# ------------------------------------ factorised MoL: routed encoders AND decoders
def install_mol_routed(model, e_enc=4, e_dec=None, scale=10.0, var_coef=0.0,
                       anneal=0, noise=0.0, tied=True, verbose=True):
    """Two independent routers per LatentKV (#54): encoder router on x, decoder router
    on the cached latent. No index. E_enc x E_dec per-token maps from E_enc + E_dec
    matrices.

    The STRUCTURE is created HERE, at install, as E copies of the plain factors --
    exact at install (every expert equals plain MLA, so any routing gives the same
    output). It must exist before the optimizer is built: parameters created later,
    at the step-0 calibration point, would never be trained. mol_routed_cluster_init()
    then fills the values IN PLACE.
    """
    from mercurius.surgery.transmla import LatentKV
    # TIED (option 1, default): decoder index = encoder index, cached at decode.
    # UNTIED (option 2): an independent decoder router on the latent -- measured on a
    # toy as +195% held-out error WITHOUT the encoder id, because each encoder writes
    # its own coordinate frame. Only sensible later, reading (e, c) not c alone.
    e_dec = e_enc if tied else (e_dec or e_enc)
    lats = [m for m in model.modules() if isinstance(m, LatentKV)]
    if not lats:
        raise SystemExit("--mla-mol-routed found no LatentKV: MLA is not installed.")
    for lat in lats:
        g = getattr(lat, "groups", None)
        # ONE group spanning every KV head is the joint latent under another name:
        # _init_grouped then runs the same whitened SVD over the same rows in the
        # same order (v_metric is refused there, as in mol_joint_W's v_mat=False).
        # The "ungrouped" JSONs are written in that form, so accept it.
        if g and len(g) > 1:
            raise SystemExit("--mla-mol-routed needs UNGROUPED latents (one per layer); "
                             "the per-cluster factorisation rebuilds the joint W.")
        d, r = lat.down.in_features, lat.down.out_features
        dev = lat.down.weight.device
        lat.down_w = nn.Parameter(lat.down.weight.detach().float()
                                  .unsqueeze(0).repeat(e_enc, 1, 1))
        lat.up_k_w = nn.Parameter(lat.up_k.weight.detach().float()
                                  .unsqueeze(0).repeat(e_dec, 1, 1))
        lat.up_v_w = nn.Parameter(lat.up_v.weight.detach().float()
                                  .unsqueeze(0).repeat(e_dec, 1, 1))
        lat.mol_enc_router = LatentRouter(d, e_enc, device=dev, mode="cosine",
                                          scale=scale, var_coef=var_coef,
                                          anneal=anneal, noise=noise)
        if not tied:
            lat.mol_dec_router = LatentRouter(r, e_dec, device=dev, mode="cosine",
                                              scale=scale, var_coef=var_coef,
                                              anneal=anneal, noise=noise)
        lat.E_enc, lat.E_dec, lat.mol_routed, lat.mol_tied = e_enc, e_dec, True, tied
        for p in (*lat.down.parameters(), *lat.up_k.parameters(),
                  *lat.up_v.parameters()):
            p.requires_grad_(False)          # superseded; not trained, not saved
    if verbose:
        if tied:
            import math as _m
            print(f"  MoL (tied pairs): {e_enc} encoder/decoder pairs on {len(lats)} "
                  f"layers; router on x; decode caches a {_m.ceil(_m.log2(e_enc))}-bit "
                  f"index per token per layer; cosine router, DeepSeek selection bias "
                  f"(no aux loss)" + (f", variance loss {var_coef:g}" if var_coef else ""),
                  flush=True)
        else:
            print(f"  MoL (untied): {e_enc} encoders x {e_dec} decoders on {len(lats)} "
                  f"layers; decoder router on the cached latent", flush=True)
    return len(lats)


def capture_kv_fisher(model, windows, verbose=True):
    """OUTPUT FISHER at every LatentKV: G = E_t[g_t g_t^T], g_t = dL/d[k_t; v_t] in
    natural units, L = summed next-token CE on real windows (#56). Second order, the
    empirical-Fisher block: dL ~= 1/2 E_t[d_t^T G d_t] for a reconstruction error d_t.
    k_norm is scale-invariant per head, so G discounts errors along k -- which the
    balanced [K;V] metric cannot.

    Captured on the model AS BUILT (latents at their plain init), i.e. the Fisher of
    the compressed model; the offline measurement used the uncompressed one. The
    caller must disable gradient checkpointing (hooks on recomputed tensors) and
    zero the parameter grads afterwards."""
    from mercurius.surgery.transmla import LatentKV
    lats = [m for m in model.modules() if isinstance(m, LatentKV)]
    G = {l: None for l in lats}
    n = {l: 0 for l in lats}
    buf = {}

    def mk(l):
        def h(mod, inp, out):
            k, v = out
            if k.requires_grad:
                k.register_hook(lambda g: buf.__setitem__((id(l), "k"), g.detach()))
                v.register_hook(lambda g: buf.__setitem__((id(l), "v"), g.detach()))
        return h
    hs = [l.register_forward_hook(mk(l)) for l in lats]
    hs.append(model.get_input_embeddings().register_forward_hook(
        lambda mod, inp, out: out if out.requires_grad else out.requires_grad_(True)))
    try:
        for s_, w in enumerate(windows):
            ids = w.reshape(1, -1).cuda()
            logits = model(input_ids=ids).logits[0, :-1].float()
            loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum")
            loss.backward()
            del logits, loss
            for l in lats:
                gk, gv = buf[(id(l), "k")], buf[(id(l), "v")]
                g = torch.cat([gk.reshape(-1, gk.shape[-1]),
                               gv.reshape(-1, gv.shape[-1])], 1).double()
                G[l] = g.t() @ g if G[l] is None else G[l] + g.t() @ g
                n[l] += g.shape[0]
            buf.clear()
            if verbose and (s_ + 1) % 64 == 0:
                print(f"    output Fisher: {s_+1}/{len(windows)} windows", flush=True)
    finally:
        for h in hs:
            h.remove()
    model.zero_grad(set_to_none=True)
    return {l: (G[l] / max(n[l], 1)).cpu() for l in lats}


@torch.no_grad()
def mol_tied_cluster_init(model, X_by_lat, G_by_lat=None, verbose=True):
    """Cluster-Aware Upcycling for TIED pairs (#55; arXiv 2604.13508).

    Spherical k-means on x (E clusters); pair e <- the whitened-SVD factorisation of
    the joint W under cluster e's covariance (lat.mol_factor: the SAME math as the
    MLA constructor, so E=1 with the global covariance reproduces plain MLA). Router
    <- normalised x-centroids, bias <- 0. A cluster with fewer than d_model rows (no
    full-rank covariance) keeps the plain factors.
    Reports IN-SAMPLE relative [K;V] error, plain -> tied pairs.
    """
    reports = []
    for lat, X in X_by_lat.items():
        if not (getattr(lat, "mol_routed", False) and getattr(lat, "mol_tied", False)):
            continue
        dev = lat.down_w.device
        Xf = X.to(dev, torch.float32)
        d_model, r = Xf.shape[1], lat.down_w.shape[1]
        Wk, Wv = (w.to(dev, torch.float32) for w in lat._W_ref)
        K, V = Xf @ Wk.t(), Xf @ Wv.t()
        nKV = K.norm() ** 2 + V.norm() ** 2
        # plain factors are FROZEN, hence NF4-packed by the time this runs; slot 0 of
        # each stack is still the exact copy install_mol_routed took of them
        cp = Xf @ lat.down_w[0].t()
        plain_up_k, plain_up_v = lat.up_k_w[0].clone(), lat.up_v_w[0].clone()
        e_plain = float(((cp @ plain_up_k.t() - K).norm() ** 2
                         + (cp @ plain_up_v.t() - V).norm() ** 2) / nKV) ** .5
        C, a_ = _spherical_kmeans(Xf, lat.E_enc)
        fit = 0
        for e in range(lat.E_enc):
            m = a_ == e
            if int(m.sum()) >= d_model:
                Xe = Xf[m].double()
                dn, uk, uv = lat.mol_factor((Xe.t() @ Xe) / Xe.shape[0], r,
                                            G=(G_by_lat or {}).get(lat))
                lat.down_w[e].copy_(dn.float())
                lat.up_k_w[e].copy_(uk.float())
                lat.up_v_w[e].copy_(uv.float())
                fit += 1
        lat.mol_enc_router.w.copy_(C)
        lat.mol_enc_router.bias.zero_()
        was = lat.mol_enc_router.training
        lat.mol_enc_router.eval()
        ei, _ = lat.mol_enc_router(Xf)
        lat.mol_enc_router.train(was)
        pk, pv = torch.empty_like(K), torch.empty_like(V)
        for e in range(lat.E_enc):
            m = ei == e
            if m.any():
                c = Xf[m] @ lat.down_w[e].t()
                pk[m], pv[m] = c @ lat.up_k_w[e].t(), c @ lat.up_v_w[e].t()
        e_tied = float(((pk - K).norm() ** 2 + (pv - V).norm() ** 2) / nKV) ** .5
        Gl = (G_by_lat or {}).get(lat)
        if Gl is not None:                      # dCE proxy, plain -> tied (nats/token)
            Gd = Gl.to(dev, torch.float32)
            def _dce(dk, dv):
                D_ = torch.cat([dk, dv], 1)
                return 0.5 * float(((D_ @ Gd) * D_).sum(1).mean())
            dce_plain = _dce(cp @ plain_up_k.t() - K, cp @ plain_up_v.t() - V)
            dce_tied = _dce(pk - K, pv - V)
        else:
            dce_plain = dce_tied = None
        agree = float((ei == a_).float().mean())
        rep = dict(r=r, e_plain=e_plain, e_tied=e_tied, fit=fit,
                   dce_plain=dce_plain, dce_tied=dce_tied,
                   sizes=torch.bincount(a_, minlength=lat.E_enc).tolist(),
                   router_agree=agree)
        reports.append(rep)
        if verbose:
            print(f"    MoL tied init r={r:>4}: err plain {e_plain:.4f} -> tied "
                  f"{e_tied:.4f} ({100*(e_tied/e_plain-1):+.1f}%)  fitted {fit}/"
                  f"{lat.E_enc}  sizes {rep['sizes']}  router/k-means agree "
                  f"{100*agree:.1f}%" + (
                      f"  | dCE proxy {1e3*dce_plain:.3f} -> {1e3*dce_tied:.3f} e-3 "
                      f"({100*(dce_tied/dce_plain-1):+.1f}%)" if dce_plain else ""),
                  flush=True)
    return reports


@torch.no_grad()
def mol_routed_cluster_init(model, X_by_lat, ridge=1e-3, verbose=True):
    """Two-stage cluster-aware init (#54), on inputs from the FULLY BUILT student.

    1. ENCODERS: spherical k-means on x (E_enc); each encoder <- the whitened-SVD
       factorisation of the joint W under ITS cluster's covariance (lat.mol_factor,
       the same math as the MLA constructor -- the Cluster-Aware Upcycling recipe,
       arXiv 2604.13508). Encoder router <- normalised x-centroids.
    2. DECODERS: encode the calibration set with the routed encoders, spherical
       k-means on the RESULTING LATENTS (E_dec), and fit each decoder by ridged least
       squares to the TRUE K/V of its latent region. Decoder router <- normalised
       latent centroids. Decoders are therefore fitted to whatever MIXTURE of encoders
       actually lands in each latent region, rather than assuming tidy pairs.
    Reports in-sample relative [K;V] error: plain -> routed encoders + one global
    decoder -> full factorised, and latent-cluster PURITY (share of each latent
    cluster coming from its dominant encoder).
    """
    reports = []
    for lat, X in X_by_lat.items():
        if not getattr(lat, "mol_routed", False):
            continue
        dev = lat.down_w.device
        Xf = X.to(dev, torch.float32)
        d_model, r = Xf.shape[1], lat.down_w.shape[1]
        Wk, Wv = (w.to(dev, torch.float32) for w in lat._W_ref)
        K, V = Xf @ Wk.t(), Xf @ Wv.t()
        nKV = K.norm() ** 2 + V.norm() ** 2

        def err(pk, pv):
            return float(((pk - K).norm() ** 2 + (pv - V).norm() ** 2) / nKV) ** 0.5

        def ls(cm, T):
            G = cm.t() @ cm
            G += ridge * float(G.diagonal().mean()) * torch.eye(r, device=dev)
            return torch.linalg.solve(G, cm.t() @ T).t()

        cp = Xf @ lat.down_w[0].t()          # slot 0 = plain copy; lat.down is NF4 now
        e_plain = err(cp @ lat.up_k_w[0].t(), cp @ lat.up_v_w[0].t())

        # ---- stage 1: encoders from x-clusters ----
        Ce, ae = _spherical_kmeans(Xf, lat.E_enc)
        enc_fit = 0
        for e in range(lat.E_enc):
            m = ae == e
            if int(m.sum()) >= d_model:             # need a full-rank covariance
                Xe = Xf[m].double()
                cov = (Xe.t() @ Xe) / Xe.shape[0]
                down, _, _ = lat.mol_factor(cov, r)
                lat.down_w[e].copy_(down.float())
                enc_fit += 1
        lat.mol_enc_router.w.copy_(Ce)
        lat.mol_enc_router.bias.zero_()
        lat.mol_enc_router.eval()
        ei, _ = lat.mol_enc_router(Xf)
        c = torch.empty(Xf.shape[0], r, device=dev)
        for e in range(lat.E_enc):
            m = ei == e
            if m.any():
                c[m] = Xf[m] @ lat.down_w[e].t()
        gk, gv = ls(c, K), ls(c, V)
        e_enc_only = err(c @ gk.t(), c @ gv.t())

        # ---- stage 2: decoders from latent clusters ----
        Cd, ad = _spherical_kmeans(c, lat.E_dec)
        dec_fit = 0
        for dd in range(lat.E_dec):
            m = ad == dd
            if int(m.sum()) >= 1.5 * r:
                lat.up_k_w[dd].copy_(ls(c[m], K[m]))
                lat.up_v_w[dd].copy_(ls(c[m], V[m]))
                dec_fit += 1
            else:
                lat.up_k_w[dd].copy_(gk)
                lat.up_v_w[dd].copy_(gv)
        lat.mol_dec_router.w.copy_(Cd)
        lat.mol_dec_router.bias.zero_()
        lat.mol_dec_router.eval()
        di, _ = lat.mol_dec_router(c)
        pk, pv = torch.empty_like(K), torch.empty_like(V)
        for dd in range(lat.E_dec):
            m = di == dd
            if m.any():
                pk[m] = c[m] @ lat.up_k_w[dd].t()
                pv[m] = c[m] @ lat.up_v_w[dd].t()
        e_full = err(pk, pv)
        pur = []
        for dd in range(lat.E_dec):
            m = di == dd
            if m.any():
                pur.append(float(torch.bincount(ei[m], minlength=lat.E_enc).max()
                                 / m.sum()))
        used = int(torch.unique(ei * lat.E_dec + di).numel())
        rep = dict(r=r, e_plain=e_plain, e_enc=e_enc_only, e_full=e_full,
                   enc_fit=enc_fit, dec_fit=dec_fit,
                   enc_sizes=torch.bincount(ae, minlength=lat.E_enc).tolist(),
                   purity=sum(pur) / max(len(pur), 1), pairs_used=used)
        reports.append(rep)
        if verbose:
            print(f"    MoL routed init r={r:>4}: err plain {e_plain:.4f} -> +enc "
                  f"{e_enc_only:.4f} -> +dec {e_full:.4f}  enc fit {enc_fit}/"
                  f"{lat.E_enc} dec fit {dec_fit}/{lat.E_dec}  purity "
                  f"{rep['purity']:.2f}  pairs used {used}/{lat.E_enc*lat.E_dec}",
                  flush=True)
    return reports
