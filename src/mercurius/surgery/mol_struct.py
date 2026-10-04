"""STRUCTURED mixtures of latents for TRAINING (#57, arm C): residual 2-stage and top-2 sum.

Offline (#57) these were fitted in closed form and scored through explicit cache codecs;
this module puts the same structures in the model so they can be TRAINED:

  resid   stage 1: one of E rank-h pieces; stage 2: one of E rank-(r-h) pieces fitted on
          what stage 1 left.  cache = r values + 2 indices;  E^2 reachable maps.
  top2    ONE dictionary of E rank-h pieces; each token uses the best PAIR, summed.
          cache = 2h values (<= r) + 2 indices;  C(E,2) reachable maps.

ROUTING, two modes (--mol-struct-routing):
  learned (DEFAULT, user 2026-09-28): one cosine LatentRouter PER GROUP on x -- the same
          router class and settings as arm B (DeepSeek selection bias, no aux loss, gate
          ratio g/g.detach() = 1.0 forward so the router gets the LOSS gradient).
          Centroids initialised from the closed-form fit's assignments; after that the
          loss moves pieces and routing together. "Best reconstructs the original W x"
          is only locally optimal and would anchor routing to the frozen W forever.
          For resid this is GROUPED TOP-2: one piece from each of two groups, summed.
  best    (analysis only): a token takes the piece(s) that best
reconstruct its ORIGINAL [k; v] = W x in the metric A the init was fitted in (output
Fisher G^1/2, or the K/V balance diagonal). It needs only x and the frozen W, so it is
computable at prefill, and the chosen indices are what the cache stores. Selection runs
under no_grad; gradients flow through the chosen pieces only. No router, no balance loss.

DEEPSEEK-V3 SELECTION BIAS (user, 2026-09-28): best-of-E has no router feedback loop,
but TRAINING has one -- a piece chosen more gets more gradient, improves, and is chosen
more. So selection minimises  err_e / mean_e(err)  -  b_e  (per-token normalised
error, so b is scale-free), with b_e += gamma * sign(mean_load - load_e) after every
optimizer step (mol_struct_update_bias), exactly the aux-loss-free rule of arm B. For
top2 the pair score subtracts b_a + b_b. b is saved and used at inference, as in V3. NOTE this differs from top-1's cosine router, so if a structure beats
top-1 the tie-breaker is top-1 with this same routing (--mla-mol-struct top1).

  top1    also available here: one of E rank-r pieces, the same routing -- the
          router-matched control for the two structures.

EXACT AT INSTALL: pieces are copies of the plain factors' row blocks, so any routing
reproduces plain MLA before mol_struct_init refits them.

CHECKPOINTS: the pieces are ordinary trainable Parameters. The metric A is a Parameter
with requires_grad=True that is only ever read under no_grad, so its grad stays None and
NAdam never updates it (the LatentRouter.bias pattern) -- that is what makes
save_trainable keep it, so a replay routes exactly as training did.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mercurius.surgery.mol import _spherical_kmeans, LatentRouter
from mercurius.surgery.transmla import LatentKV, whiten_factor


# ------------------------------------------------------------------------------
# closed-form fitting -- same math as experiments/mol_structures_fit.py (#57)
# ------------------------------------------------------------------------------
class Metric:
    def __init__(self, A):
        self.A = A

    def sq(self, D):
        return ((D @ self.A.t()) ** 2).sum(1)


def rrr(X, R, A, Ai, r):
    """min_{rank<=r} E||A (R - B x)||^2 -> (down (r,d), up (out,r))."""
    n = X.shape[0]
    Xd = X.double()
    S = (Xd.t() @ Xd / n).float()
    C = (R.double().t() @ Xd / n).float()
    L = whiten_factor(S)
    Linv = torch.linalg.solve_triangular(
        L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    U, s, Vh = torch.linalg.svd(A @ C @ Linv.t(), full_matrices=False)
    return Vh[:r] @ Linv, Ai @ (U[:, :r] * s[:r])


def _recon(X, pieces, a, bs=16384):
    out = torch.empty(X.shape[0], pieces[0][1].shape[0], device=X.device)
    for e, (d, u) in enumerate(pieces):
        idx = (a == e).nonzero().squeeze(1)
        for s in range(0, idx.numel(), bs):
            j = idx[s:s + bs]
            out[j] = X[j] @ d.t() @ u.t()
    return out


def _best(X, R, pieces, M, bs=8192):
    out = []
    for s in range(0, X.shape[0], bs):
        x, t = X[s:s + bs], R[s:s + bs]
        out.append(torch.stack([M.sq(x @ d.t() @ u.t() - t) for d, u in pieces], 1)
                   .argmin(1))
    return torch.cat(out)


def _ksub(X, R, E, r, A, Ai, M, iters):
    glob = rrr(X, R, A, Ai, r)
    if E == 1:
        return [glob]
    _, a = _spherical_kmeans(X, E)
    for it in range(iters + 1):
        pieces = [rrr(X[a == e], R[a == e], A, Ai, r)
                  if int((a == e).sum()) >= X.shape[1] else glob for e in range(E)]
        if it < iters:
            a = _best(X, R, pieces, M)
    return pieces


def fit_resid(X, T, spec, A, Ai, M, iters, backfit=1):
    """spec [(E1, r1), (E2, r2)]: greedy on residuals, then backfitted."""
    stages, recs = [], []
    for E, r in spec:
        R = T - (sum(recs) if recs else 0)
        p = _ksub(X, R, E, r, A, Ai, M, iters)
        stages.append(p)
        recs.append(_recon(X, p, _best(X, R, p, M) if E > 1 else
                           torch.zeros(X.shape[0], dtype=torch.long, device=X.device)))
    for _ in range(backfit):
        for k, (E, r) in enumerate(spec):
            others = sum(rc for j, rc in enumerate(recs) if j != k)
            p = _ksub(X, T - others, E, r, A, Ai, M, iters)
            stages[k] = p
            recs[k] = _recon(X, p, _best(X, T - others, p, M) if E > 1 else
                             torch.zeros(X.shape[0], dtype=torch.long, device=X.device))
    return stages


def pair_assign(X, T, pieces, M, bs=2048):
    """Exact best unordered pair per token via the per-token Gram in the metric."""
    E = len(pieces)
    iu = torch.triu_indices(E, E, 1, device=X.device)
    o1, o2 = [], []
    for s in range(0, X.shape[0], bs):
        x, t = X[s:s + bs], T[s:s + bs]
        RA = torch.stack([(x @ d.t() @ u.t()) @ M.A.t() for d, u in pieces], 1)
        TA = t @ M.A.t()
        G = RA @ RA.transpose(1, 2)
        c = (RA * TA[:, None, :]).sum(-1)
        dg = G.diagonal(dim1=1, dim2=2)
        err = dg[:, iu[0]] + dg[:, iu[1]] + 2 * G[:, iu[0], iu[1]] \
            - 2 * c[:, iu[0]] - 2 * c[:, iu[1]]
        k = err.argmin(1)
        o1.append(iu[0][k]); o2.append(iu[1][k])
    return torch.cat(o1), torch.cat(o2)


def fit_top2(X, T, E, h, A, Ai, M, iters):
    """Dictionary = resid-(E/2)x(E/2) stage pieces (so top2 <= resid at equal params),
    then exact pair assignment / Gauss-Seidel piece refits on partner residuals."""
    st = fit_resid(X, T, [(E // 2, h), (E - E // 2, h)], A, Ai, M, iters)
    pieces = list(st[0]) + list(st[1])
    for it in range(iters + 1):
        a1, a2 = pair_assign(X, T, pieces, M)
        if it == iters:
            break
        for e in range(E):
            m1, m2 = a1 == e, a2 == e
            idx = (m1 | m2).nonzero().squeeze(1)
            if idx.numel() < X.shape[1]:
                continue
            other = torch.where(m1[idx], a2[idx], a1[idx])
            pieces[e] = rrr(X[idx], T[idx] - _recon(X[idx], pieces, other), A, Ai, h)
    return pieces


# ------------------------------------------------------------------------------
# the module state on a LatentKV
# ------------------------------------------------------------------------------
def _stacks(lat, prefix, E, r, d, dev):
    """(E, r, d) down and (E, out, r) up_k / up_v, as float32 Parameters."""
    setattr(lat, f"{prefix}_down", nn.Parameter(torch.zeros(E, r, d, device=dev)))
    setattr(lat, f"{prefix}_up_k", nn.Parameter(torch.zeros(E, lat.k_out, r, device=dev)))
    setattr(lat, f"{prefix}_up_v", nn.Parameter(torch.zeros(E, lat.v_out, r, device=dev)))


def _stage_names(lat):
    return {"resid": ["ms1", "ms2"], "top1": ["ms1"], "top2": ["msd"]}[lat.mol_struct]


def install_mol_struct(model, kind, E, routing="learned", scale=10.0, verbose=True):
    """kind in {resid, top2, top1}; routing in {learned, best}. Exact at install
    (copies of plain's row blocks), whatever the routing."""
    if routing == "learned" and kind == "top2":
        raise SystemExit("learned routing for FLAT top-2 is not implemented: a cosine "
                         "top-2 over one dictionary can pick two redundant pieces at "
                         "init. Use --mla-mol-struct resid (GROUPED top-2: one learned "
                         "router per group), which is what the fitted top-2 reduces to.")
    lats = [m for m in model.modules() if isinstance(m, LatentKV)]
    if not lats:
        raise SystemExit("--mla-mol-struct found no LatentKV: MLA is not installed.")
    for lat in lats:
        g = getattr(lat, "groups", None)
        if g and len(g) > 1:
            raise SystemExit("--mla-mol-struct needs one latent per layer.")
        d, r = lat.down.in_features, lat.down.out_features
        dev = lat.down.weight.device
        Dn = lat.down.weight.detach().float()
        Uk = lat.up_k.weight.detach().float()
        Uv = lat.up_v.weight.detach().float()
        h = r // 2
        if kind == "resid":
            _stacks(lat, "ms1", E, h, d, dev)
            _stacks(lat, "ms2", E, r - h, d, dev)
            blocks = {"ms1": slice(0, h), "ms2": slice(h, r)}
        elif kind == "top1":
            _stacks(lat, "ms1", E, r, d, dev)
            blocks = {"ms1": slice(0, r)}
        elif kind == "top2":
            _stacks(lat, "msd", E, h, d, dev)
            blocks = None
        else:
            raise ValueError(kind)
        with torch.no_grad():
            if blocks:
                for pre, sl in blocks.items():
                    getattr(lat, f"{pre}_down").copy_(Dn[sl].expand(E, -1, -1))
                    getattr(lat, f"{pre}_up_k").copy_(Uk[:, sl].expand(E, -1, -1))
                    getattr(lat, f"{pre}_up_v").copy_(Uv[:, sl].expand(E, -1, -1))
            else:        # top2: half the pieces hold plain's first h rows, half the next
                for e in range(E):
                    sl = slice(0, h) if e < E // 2 else slice(h, 2 * h)
                    lat.msd_down[e].copy_(Dn[sl]); lat.msd_up_k[e].copy_(Uk[:, sl])
                    lat.msd_up_v[e].copy_(Uv[:, sl])
        # routing metric: K/V balance diagonal until mol_struct_init sets the fitted one
        out = lat.k_out + lat.v_out
        diag = torch.cat([torch.full((lat.k_out,), float(lat.k_scale)),
                          torch.full((lat.v_out,), float(lat.v_scale))]).to(dev)
        lat.ms_metric = nn.Parameter(torch.diag(diag))    # grad stays None (no_grad use)
        for pre in {"resid": ["ms1", "ms2"], "top1": ["ms1"], "top2": ["msd"]}[kind]:
            setattr(lat, f"{pre}_bias", nn.Parameter(torch.zeros(E, device=dev)))
            if routing == "learned":
                setattr(lat, f"{pre}_router",
                        LatentRouter(d, E, device=dev, mode="cosine", scale=scale))
        lat.ms_routing = routing
        lat._ms_AW = None
        lat.mol_struct, lat.ms_E = kind, E
        for p in (*lat.down.parameters(), *lat.up_k.parameters(), *lat.up_v.parameters()):
            p.requires_grad_(False)
    if verbose:
        per = {"resid": "E x E = %d maps, 2 indices" % (E * E),
               "top1": "%d maps, 1 index" % E,
               "top2": "C(E,2) = %d maps, 2 indices" % (E * (E - 1) // 2)}[kind]
        how = ("LEARNED cosine router per group + DeepSeek bias (as arm B)"
               if routing == "learned" else
               "ANALYTICAL best-of-E on the original W x (analysis only)")
        print(f"  MoL-struct ({kind}): E={E} on {len(lats)} layers; {per}; {how}",
              flush=True)
    return len(lats)


def _AW(lat, dtype):
    """A @ W, cached; W from the frozen reference, A never changes after init."""
    if lat._ms_AW is None or lat._ms_AW.dtype != dtype:
        Wk, Wv = (w.to(lat.ms_metric.device, torch.float32) for w in lat._W_ref)
        lat._ms_AW = (lat.ms_metric.detach().float() @ torch.cat([Wk, Wv], 0)).to(dtype)
    return lat._ms_AW


def mol_struct_forward(lat, x):
    """Returns (k, v) shaped like x[..., :] -> (..., k_out) / (..., v_out)."""
    xf = x.reshape(-1, x.shape[-1])
    dt = xf.dtype
    names = _stage_names(lat)
    if getattr(lat, "ms_routing", "best") == "learned":
        picks, gates = [], []
        for pre in names:
            idx, g = getattr(lat, f"{pre}_router")(xf)
            picks.append((pre, idx.reshape(-1)))
            gates.append(g.reshape(-1))
        return _decode(lat, x, xf, picks, gates)
    A = lat.ms_metric.detach().to(dt)
    with torch.no_grad():
        TA = xf @ _AW(lat, dt).t()                             # target in the metric

        def cand(pre):                                         # (N, E, out) in metric
            D = getattr(lat, f"{pre}_down").detach().to(dt)
            U = torch.cat([getattr(lat, f"{pre}_up_k"), getattr(lat, f"{pre}_up_v")],
                          1).detach().to(dt)
            AU = A @ U                                         # (E, out, h)
            c = torch.einsum("nd,ehd->neh", xf, D)
            return torch.einsum("neh,eoh->neo", c, AU)

        if lat.mol_struct == "top2":
            RA = cand("msd")
            G = RA @ RA.transpose(1, 2)
            cc = (RA * TA[:, None, :]).sum(-1)
            E = RA.shape[1]
            iu = torch.triu_indices(E, E, 1, device=xf.device)
            dg = G.diagonal(dim1=1, dim2=2)
            # + ||t||^2 so the pair error is the true squared error (>= 0) and can
            # be normalised per token for the scale-free bias
            tt = (TA.float() ** 2).sum(-1, keepdim=True)
            err = (dg[:, iu[0]] + dg[:, iu[1]] + 2 * G[:, iu[0], iu[1]]
                   - 2 * cc[:, iu[0]] - 2 * cc[:, iu[1]]).float() + tt
            b = lat.msd_bias.detach().float()
            err = err / err.mean(1, keepdim=True).clamp_min(1e-12) - (b[iu[0]] + b[iu[1]])
            kk = err.argmin(1)
            picks = [("msd", iu[0][kk]), ("msd", iu[1][kk])]
            del RA, G
        else:
            picks, left = [], TA
            for pre in names:
                RA = cand(pre)
                err = ((left[:, None, :] - RA).float() ** 2).sum(-1)
                b = getattr(lat, f"{pre}_bias").detach().float()
                a = (err / err.mean(1, keepdim=True).clamp_min(1e-12) - b).argmin(1)
                left = left - RA[torch.arange(RA.shape[0], device=xf.device), a]
                picks.append((pre, a))
                del RA
    return _decode(lat, x, xf, picks, None)


def _decode(lat, x, xf, picks, gates):
    """Sum the chosen pieces' reconstructions. gates (learned routing): per-token
    g/g.detach(), exactly 1.0 in value, carrying the loss gradient to the router."""
    dt = xf.dtype
    k = xf.new_zeros(xf.shape[0], lat.k_out)
    v = xf.new_zeros(xf.shape[0], lat.v_out)
    for s_, (pre, a) in enumerate(picks):
        D = getattr(lat, f"{pre}_down"); Uk = getattr(lat, f"{pre}_up_k")
        Uv = getattr(lat, f"{pre}_up_v")
        for e in range(D.shape[0]):
            sel = (a == e).nonzero().squeeze(1)
            if sel.numel():
                c = xf[sel] @ D[e].t().to(dt)
                if gates is not None:
                    c = c * gates[s_][sel].unsqueeze(1).to(dt)
                k = k.index_add(0, sel, (c @ Uk[e].t().to(dt)).to(k.dtype))
                v = v.index_add(0, sel, (c @ Uv[e].t().to(dt)).to(v.dtype))
    lat._ms_last = [a for _, a in picks]
    # per-stage piece loads for the bias step (top2: both slots count for msd)
    E_ = lat.ms_E
    loads = {}
    for pre, a in picks:
        cnt = torch.bincount(a, minlength=E_).float()
        loads[pre] = loads[pre] + cnt if pre in loads else cnt
    lat._ms_loads = loads
    return (k.reshape(*x.shape[:-1], lat.k_out), v.reshape(*x.shape[:-1], lat.v_out))


@torch.no_grad()
def mol_struct_update_bias(model, gamma=0.01):
    """b_e += gamma * sign(mean_load - load_e) per stage, once per optimizer step.
    gamma is in units of per-token NORMALISED error (typical best-vs-second gaps
    ~0.05-0.3), hence smaller than arm B's 0.03 on cosine logits. Returns max/mean."""
    worst = 0.0
    for lat in model.modules():
        if not isinstance(lat, LatentKV) or not getattr(lat, "mol_struct", None):
            continue
        for pre, load in getattr(lat, "_ms_loads", {}).items():
            b = getattr(lat, f"{pre}_bias")
            load = load.to(b.device)
            if float(load.sum()) == 0:
                continue
            b.data.add_(gamma * torch.sign(load.mean() - load))
            worst = max(worst, float(load.max() / load.mean().clamp_min(1e-9)))
    return worst


@torch.no_grad()
def _fwd_chunked(lat, X, bs=8192):
    ks, vs, loads = [], [], None
    for s in range(0, X.shape[0], bs):
        k, v = mol_struct_forward(lat, X[s:s + bs])
        ks.append(k.float()); vs.append(v.float())
        cur = [torch.bincount(a, minlength=lat.ms_E) for a in lat._ms_last]
        loads = cur if loads is None else [x + y for x, y in zip(loads, cur)]
    return torch.cat(ks), torch.cat(vs), [l.tolist() for l in loads]


@torch.no_grad()
def mol_struct_init(model, X_by_lat, G_by_lat=None, iters=2, damp=1e-3, verbose=True):
    """Closed-form init on calibration inputs from the built student (#57), in the
    output-Fisher metric when G is given, else the K/V balance diagonal."""
    for lat, X in X_by_lat.items():
        if not getattr(lat, "mol_struct", None):
            continue
        dev = lat.ms1_down.device if hasattr(lat, "ms1_down") else lat.msd_down.device
        Xf = X.to(dev, torch.float32)
        Wk, Wv = (w.to(dev, torch.float32) for w in lat._W_ref)
        W = torch.cat([Wk, Wv], 0)
        T = Xf @ W.t()
        G = (G_by_lat or {}).get(lat)
        if G is not None:
            ev, Q = torch.linalg.eigh(G.to("cpu", torch.float64))
            ev = ev.clamp_min(0) + damp * float(ev.clamp_min(0).mean())
            A = ((Q * ev.sqrt()) @ Q.T).to(dev, torch.float32)
            Ai = ((Q * ev.rsqrt()) @ Q.T).to(dev, torch.float32)
        else:
            A = lat.ms_metric.detach().float()
            Ai = torch.diag(1.0 / A.diagonal())
        M = Metric(A)
        E, r = lat.ms_E, lat.down.out_features
        h = r // 2
        # plain reference = the install copy (any routing reproduces plain exactly)
        k0, v0, _ = _fwd_chunked(lat, Xf.to(torch.bfloat16))
        e_plain = float(M.sq(torch.cat([k0, v0], -1) - T).sum() / M.sq(T).sum()) ** .5
        if lat.mol_struct == "resid":
            st = fit_resid(Xf, T, [(E, h), (E, r - h)], A, Ai, M, iters)
            fills = {"ms1": st[0], "ms2": st[1]}
        elif lat.mol_struct == "top1":
            fills = {"ms1": _ksub(Xf, T, E, r, A, Ai, M, iters)}
        else:
            fills = {"msd": fit_top2(Xf, T, E, h, A, Ai, M, iters)}
        for pre, pieces in fills.items():
            for e, (dn, up) in enumerate(pieces):
                getattr(lat, f"{pre}_down")[e].copy_(dn)
                getattr(lat, f"{pre}_up_k")[e].copy_(up[:lat.k_out])
                getattr(lat, f"{pre}_up_v")[e].copy_(up[lat.k_out:])
        lat.ms_metric.data.copy_(A)
        lat._ms_AW = None
        learned = getattr(lat, "ms_routing", "best") == "learned"
        lat.ms_routing = "best"                       # fitted assignment first
        k1, v1, loads = _fwd_chunked(lat, Xf.to(torch.bfloat16))
        e_st = float(M.sq(torch.cat([k1, v1], -1) - T).sum() / M.sq(T).sum()) ** .5
        extra = ""
        if learned:
            # router centroids <- normalised mean input of each piece's fitted tokens
            names = _stage_names(lat)
            sums = {pre: torch.zeros(E, Xf.shape[1], device=dev) for pre in names}
            best_idx = {pre: [] for pre in names}
            for s0 in range(0, Xf.shape[0], 8192):
                xb = Xf[s0:s0 + 8192]
                mol_struct_forward(lat, xb.to(torch.bfloat16))
                xn = F.normalize(xb, dim=-1)
                for pre, a in zip(names, lat._ms_last):
                    sums[pre].index_add_(0, a, xn)
                    best_idx[pre].append(a)
            for pre in names:
                rt = getattr(lat, f"{pre}_router")
                cen = F.normalize(sums[pre], dim=-1)
                empty = sums[pre].norm(dim=-1) == 0
                if empty.any():                    # unused piece: keep a random centroid
                    cen[empty] = F.normalize(torch.randn_like(cen[empty]), dim=-1)
                rt.w.data.copy_(cen)
                rt.bias.data.zero_()
            lat.ms_routing = "learned"
            was = [getattr(lat, f"{pre}_router").training for pre in names]
            for pre in names:
                getattr(lat, f"{pre}_router").eval()
            k2, v2, _ = _fwd_chunked(lat, Xf.to(torch.bfloat16))
            e_lr = float(M.sq(torch.cat([k2, v2], -1) - T).sum() / M.sq(T).sum()) ** .5
            # agreement of the learned router with the fitted assignment, per group
            ag = []
            for pre in names:
                bi = torch.cat(best_idx[pre])
                li = []
                for s0 in range(0, Xf.shape[0], 8192):
                    li.append(getattr(lat, f"{pre}_router")(Xf[s0:s0 + 8192])[0].reshape(-1))
                ag.append(float((torch.cat(li) == bi).float().mean()))
            for pre, w_ in zip(names, was):
                getattr(lat, f"{pre}_router").train(w_)
            extra = (f" | learned router: err {e_lr:.4f} ({100*(e_lr/e_plain-1):+.1f}% vs "
                     f"plain), agrees with fit {', '.join(f'{100*g:.0f}%' for g in ag)}")
        else:
            lat.ms_routing = "best"
        if verbose:
            print(f"    MoL-struct init {lat.mol_struct} r={r:>4}: err ({'Fisher' if G is not None else 'balanced'}"
                  f" metric, in-sample) plain {e_plain:.4f} -> fit {e_st:.4f} "
                  f"({100*(e_st/e_plain-1):+.1f}%){extra}", flush=True)
