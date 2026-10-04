"""Which LATENT STRUCTURE beats top-1 MoL's log(E) scaling at EQUAL CACHE? (Fisher metric)

WHY (2026-09-28). Top-1 MoL with E pieces reaches E subspaces; measured held-out error
falls only ~2-3% per doubling of E while parameters grow linearly in E. The only
structures that change that exchange rate are COMBINATORIAL (several smaller choices
combined: residual / additive subspace coding, cf. residual VQ). A DeepSeek-style
shared expert adds NO capacity at fixed cache (it is a constrained special case of
top-1: each cluster's best rank-r map is already its own SVD) -- but it cuts
parameters, so it is measured too, per parameter.

Everything is fitted in closed form, in the OUTPUT-FISHER metric (mol_fisher_fit.py):
one solver, reduced-rank regression

    min_{rank B <= r}  E || A (R - B x) ||^2     R any target (W x, or a residual)
    B* = C S^-1 (C = E[R x^T], S = E[x x^T] = L L^T);  SVD of A C L^-T -> top r
    up = A^-1 U_r S_r,  down = V_r^T L^-1

which reduces to the whitened SVD of A W L when R = W x (the MLA / MoL fit).
Pieces are chosen per token by BEST-OF-E under the Fisher metric (uses only x and the
fixed W, G: deployable at prefill); k-subspaces alternation from a spherical k-means
start. Residual stages are fitted greedily and then backfitted once.

VARIANTS, all at the layer's cache r0 (values) + index bits:
  plain            one RRR, rank r0 (and a rank sweep to price everything in rank)
  top1-E           E pieces, rank r0                      E subspaces      E*r0 params
  shared+E         global rank r0/2  +  1 of E at r0/2     E                (1+E)*r0/2
  resid-E1xE2      1 of E1 at r0/2, then 1 of E2 on the residual at r0/2
                                                           E1*E2            (E1+E2)*r0/2
  top2sum-E        ONE dictionary of E pieces at r0/2; each token takes the best PAIR
                   and sums them                       C(E,2) subspaces   E*r0/2 params
                   (top2sum-16 has top1-8's parameters with 120 vs 8 subspaces).
                   Replaced the K/V-split variant at the user's request.
REPORTED per variant: held-out Fisher error, dCE proxy, the EQUIVALENT PLAIN RANK
(interpolated on the plain sweep) -> cache saving at equal quality, params, bits.

    .venv/bin/python experiments/mol_structures_fit.py --layers 3 7 19 31
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from mercurius.models.kda import load_kda_model
from mercurius.paths import FINEWEB_LONG, STAGE_AB, WIKITEXT
from mercurius.surgery.mol import _spherical_kmeans
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.transmla import whiten_factor

sys.path.insert(0, os.path.dirname(__file__))
from mol_offline_fit import windows, capture                       # noqa: E402
from mol_fisher_fit import capture_fisher, sqrt_pair, Metric       # noqa: E402


def rrr(X, R, A, Ai, rmax):
    """Reduced-rank regression of R on X in output metric A. Returns r -> (down, up)."""
    n = X.shape[0]
    Xd = X.double()
    S = (Xd.t() @ Xd / n).float()
    C = (R.double().t() @ Xd / n).float()                       # (out, d)
    L = whiten_factor(S)
    Linv = torch.linalg.solve_triangular(
        L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    M = A @ C @ Linv.t()                                        # A C L^-T
    U, s, Vh = torch.linalg.svd(M, full_matrices=False)
    D = Vh[:rmax] @ Linv
    Up = Ai @ (U[:, :rmax] * s[:rmax])
    return lambda r: (D[:r], Up[:, :r])


def recon(X, pieces, a, bs=16384):
    """Per-token reconstruction with piece a[t]; pieces: list of (down, up)."""
    out = torch.empty(X.shape[0], pieces[0][1].shape[0], device=X.device)
    for e, (d, u) in enumerate(pieces):
        idx = (a == e).nonzero().squeeze(1)
        for s in range(0, idx.numel(), bs):
            j = idx[s:s + bs]
            out[j] = X[j] @ d.t() @ u.t()
    return out


def best_assign(X, R, pieces, M, bs=8192):
    out = []
    for s in range(0, X.shape[0], bs):
        x, t = X[s:s + bs], R[s:s + bs]
        out.append(torch.stack([M.sq(x @ d.t() @ u.t() - t) for d, u in pieces], 1)
                   .argmin(1))
    return torch.cat(out)


def ksub(X, R, E, r, A, Ai, M, iters, fallback):
    """k-subspaces on target R: k-means start, alternate best-assign / per-piece RRR."""
    if E == 1:
        return [fallback(r) if fallback else rrr(X, R, A, Ai, r)(r)], \
            torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    _, a = _spherical_kmeans(X, E)
    glob = rrr(X, R, A, Ai, r)(r)
    for it in range(iters + 1):
        pieces = [rrr(X[a == e], R[a == e], A, Ai, r)(r)
                  if int((a == e).sum()) >= X.shape[1] else glob for e in range(E)]
        if it < iters:
            a = best_assign(X, R, pieces, M)
    return pieces, a


class Stage:
    def __init__(self, pieces, rank):
        self.pieces, self.rank = pieces, rank


def apply_stages(X, T, stages, M):
    """Greedy test-time routing through residual stages; returns total reconstruction."""
    tot = torch.zeros_like(T)
    for st in stages:
        a = best_assign(X, T - tot, st.pieces, M) if len(st.pieces) > 1 else \
            torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        tot = tot + recon(X, st.pieces, a)
    return tot


def fit_residual(X, T, spec, A, Ai, M, iters, backfit=1):
    """spec: [(E1, r1), (E2, r2), ...] fitted greedily on residuals, then backfitted."""
    stages, recs = [], []
    for E, r in spec:
        R = T - (sum(recs) if recs else 0)
        p, a = ksub(X, R, E, r, A, Ai, M, iters, None)
        stages.append(Stage(p, r))
        recs.append(recon(X, p, a))
    for _ in range(backfit):
        for k, (E, r) in enumerate(spec):
            others = sum(rc for j, rc in enumerate(recs) if j != k)
            p, a = ksub(X, T - others, E, r, A, Ai, M, iters, None)
            stages[k] = Stage(p, r)
            recs[k] = recon(X, p, a)
    return stages


def pair_assign(X, T, pieces, M, bs=2048):
    """Best unordered PAIR (e1 < e2) per token for the summed reconstruction, exact:
    ||A(r1 + r2 - t)||^2 = G11 + G22 + 2 G12 - 2 c1 - 2 c2 + const, from the per-token
    Gram of the E piece reconstructions in the metric. Uses only x and fixed W."""
    E = len(pieces)
    iu = torch.triu_indices(E, E, 1, device=X.device)
    A = M.A
    o1, o2 = [], []
    for s in range(0, X.shape[0], bs):
        x, t = X[s:s + bs], T[s:s + bs]
        RA = torch.stack([(x @ d.t() @ u.t()) @ A.t() for d, u in pieces], 1)   # (b,E,out)
        TA = t @ A.t()
        G = RA @ RA.transpose(1, 2)
        c = (RA * TA[:, None, :]).sum(-1)
        dg = G.diagonal(dim1=1, dim2=2)
        err = dg[:, iu[0]] + dg[:, iu[1]] + 2 * G[:, iu[0], iu[1]] \
            - 2 * c[:, iu[0]] - 2 * c[:, iu[1]]
        k = err.argmin(1)
        o1.append(iu[0][k]); o2.append(iu[1][k])
    return torch.cat(o1), torch.cat(o2)


def fit_top2(X, T, E, h, A, Ai, M, iters):
    """Top-2 SUM from ONE dictionary of E rank-h pieces, then alternate exact pair
    assignment / Gauss-Seidel refit of each piece on its tokens' residual after
    removing the partner piece (closed-form RRR).

    INIT FROM THE RESIDUAL MODEL. The first version started every piece from a top-1
    fit at rank h, so all pieces held the SAME dominant directions, every pairwise sum
    was redundant, and it scored WORSE THAN PLAIN (x1.025 on layer 3) -- impossible for
    a well-fitted top-2, since two pieces = the top and next h directions of the plain
    SVD reproduce plain exactly. Now: fit resid-(E/2)x(E/2) and take stage-1 U stage-2
    pieces as the dictionary. The cross pairs (a in stage 1, b in stage 2) reproduce
    the residual model exactly, so top2sum-E <= resid-(E/2)x(E/2) at EQUAL params
    before any refinement; the other C(E,2) - (E/2)^2 pairs are free upside."""
    st = fit_residual(X, T, [(E // 2, h), (E - E // 2, h)], A, Ai, M, iters)
    pieces = list(st[0].pieces) + list(st[1].pieces)
    d = X.shape[1]
    for it in range(iters + 1):
        a1, a2 = pair_assign(X, T, pieces, M)
        if it == iters:
            break
        for e in range(E):
            m1, m2 = a1 == e, a2 == e
            idx = (m1 | m2).nonzero().squeeze(1)
            if idx.numel() < d:
                continue
            other = torch.where(m1[idx], a2[idx], a1[idx])
            Ro = recon(X[idx], pieces, other)
            pieces[e] = rrr(X[idx], T[idx] - Ro, A, Ai, h)(h)
    return pieces


def apply_top2(X, T, pieces, M):
    a1, a2 = pair_assign(X, T, pieces, M)
    return recon(X, pieces, a1) + recon(X, pieces, a2)


# ---------------------------------------------------------------------------------
# EXPLICIT CACHE CODECS. Every variant is scored as decode(encode(x)): encode sees x
# (and the fixed W, G -- available at prefill) and returns ONLY what is cached, a latent
# of <= r0 values plus integer indices; decode receives the cached tensors and the UP
# projections, never x or the downs. So a variant cannot score well by reading
# anything the cache does not hold.
# ---------------------------------------------------------------------------------
def encode_stages(X, T, stages, M):
    lat, ids, tot = [], [], torch.zeros_like(T)
    for st in stages:
        a = best_assign(X, T - tot, st.pieces, M) if len(st.pieces) > 1 else \
            torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        c = torch.empty(X.shape[0], st.pieces[0][0].shape[0], device=X.device)
        for e, (dn, up) in enumerate(st.pieces):
            j = (a == e).nonzero().squeeze(1)
            if j.numel():
                c[j] = X[j] @ dn.t()
                tot[j] += c[j] @ up.t()           # encoder-side running reconstruction
        lat.append(c); ids.append(a)
    return torch.cat(lat, 1), torch.stack(ids, 1)


def decode_stages(lat, ids, ups_per_stage):
    out, col = None, 0
    for s, ups in enumerate(ups_per_stage):
        w = ups[0].shape[1]
        c = lat[:, col:col + w]; col += w
        y = torch.empty(lat.shape[0], ups[0].shape[0], device=lat.device)
        for e, up in enumerate(ups):
            j = (ids[:, s] == e).nonzero().squeeze(1)
            if j.numel():
                y[j] = c[j] @ up.t()
        out = y if out is None else out + y
    return out


def encode_top2(X, T, pieces, M):
    a1, a2 = pair_assign(X, T, pieces, M)
    h = pieces[0][0].shape[0]
    c = torch.empty(X.shape[0], 2 * h, device=X.device)
    for e, (dn, _) in enumerate(pieces):
        j = (a1 == e).nonzero().squeeze(1)
        if j.numel():
            c[j, :h] = X[j] @ dn.t()
        j = (a2 == e).nonzero().squeeze(1)
        if j.numel():
            c[j, h:] = X[j] @ dn.t()
    return c, torch.stack([a1, a2], 1)


def decode_top2(lat, ids, ups):
    h = ups[0].shape[1]
    return decode_stages(lat, ids, [ups, ups]) if lat.shape[1] == 2 * h else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 7, 19, 31])
    ap.add_argument("--groups", default="cache/mla_groups_ungrouped_2048.json")
    ap.add_argument("--fit-windows", type=int, default=256)
    ap.add_argument("--fisher-windows", type=int, default=256)
    ap.add_argument("--test-windows", type=int, default=64)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--damp", type=float, default=1e-3)
    ap.add_argument("--sweep", type=float, nargs="+",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--variants", nargs="+",
                    default=["top1", "shared", "resid", "top2sum"],
                    help="families to fit; plain + the rank sweep always run")
    ap.add_argument("--top2-experts", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--out", default="logs/mol_structures_fit.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    m = load_kda_model(str(STAGE_AB), dtype=torch.bfloat16)
    for l in get_trunk(m).layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    m.eval()
    trunk = get_trunk(m)
    idx = [i for i in a.layers if hasattr(trunk.layers[i], "self_attn")]
    r0 = {int(k): sum(int(r) for _, r in v)
          for k, v in json.load(open(a.groups))["groups"].items()}

    size = FINEWEB_LONG.stat().st_size
    t0 = time.time()
    fit_w = windows(tok, FINEWEB_LONG, a.fit_windows, 0, size // 2, 1)
    Xfit = capture(m, fit_w, idx)
    Xte = {"fineweb2": capture(m, windows(tok, FINEWEB_LONG, a.test_windows,
                                          size // 2, size, 2), idx),
           "wikitext": capture(m, windows(tok, WIKITEXT, a.test_windows, 0,
                                          WIKITEXT.stat().st_size, 3), idx)}
    Gs = capture_fisher(m, fit_w[:a.fisher_windows], idx)
    print(f"inputs + output Fisher captured in {time.time()-t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

    res = {}
    for i in idx:
        t1 = time.time()
        sa = trunk.layers[i].self_attn
        Wk = sa.k_proj.weight.detach().float()
        Wv = sa.v_proj.weight.detach().float()
        W = torch.cat([Wk, Wv])
        nk, nout, d = Wk.shape[0], W.shape[0], W.shape[1]
        Gh, Ghi = sqrt_pair(Gs[i], a.damp)
        Mf = Metric(Gh)
        Mraw = Metric(sqrt_pair(Gs[i], 0.0)[0])
        X = Xfit[i].cuda().float()
        T = X @ W.t()
        R0 = r0[i]
        tests = {k: (v[i].cuda().float(), v[i].cuda().float() @ W.t()) for k, v in Xte.items()}

        def score(enc, dec, budget=None):
            """enc(Xt, Tt) -> (latent, ids); dec(latent, ids) -> [K;V]. Cache-only decode."""
            out = {}
            for k, (Xt, Tt) in tests.items():
                lat, ids = enc(Xt, Tt)
                assert lat.shape[1] <= (budget or R0), (lat.shape, R0)
                P = dec(lat, ids)
                Dl = P - Tt
                out[k] = {"fis": math.sqrt(float(Mf.sq(Dl).sum()) / float(Mf.sq(Tt).sum())),
                          "dCE": 0.5 * float(Mraw.sq(Dl).mean())}
            return out

        row = {"r0": R0, "variants": {}}
        # ---- plain rank sweep (prices everything in rank) ----
        sweep = sorted({max(1, min(nout, round(f * R0))) for f in a.sweep})
        pf = rrr(X, T, Gh, Ghi, max(sweep))
        row["sweep"] = {"ranks": sweep, "err": {}}
        for k, (Xt, Tt) in tests.items():
            row["sweep"]["err"][k] = [
                math.sqrt(float(Mf.sq(Xt @ pf(r)[0].t() @ pf(r)[1].t() - Tt).sum())
                          / float(Mf.sq(Tt).sum())) for r in sweep]

        def add(name, sc, params, bits):
            row["variants"][name] = {"score": sc, "params": params, "bits": bits}

        pdn, pup = pf(R0)
        add("plain", score(lambda Xt, Tt: (Xt @ pdn.t(), None),
                           lambda c, ids: c @ pup.t()),
            R0 * (d + nout), 0)

        h = R0 // 2
        codec = lambda st: (lambda Xt, Tt, st=st: encode_stages(Xt, Tt, st, Mf),
                            lambda c, ids, st=st: decode_stages(
                                c, ids, [[u for _, u in x.pieces] for x in st]))
        for E in (8, 16):
            if "top1" in a.variants:
                st = fit_residual(X, T, [(E, R0)], Gh, Ghi, Mf, a.iters, backfit=0)
                add(f"top1-{E}", score(*codec(st)),
                    E * R0 * (d + nout), math.ceil(math.log2(E)))
            if "shared" in a.variants:
                st = fit_residual(X, T, [(1, h), (E, R0 - h)], Gh, Ghi, Mf, a.iters)
                add(f"shared+{E}", score(*codec(st)),
                    (h + E * (R0 - h)) * (d + nout), math.ceil(math.log2(E)))
            if "resid" in a.variants:
                st = fit_residual(X, T, [(E, h), (E, R0 - h)], Gh, Ghi, Mf, a.iters)
                add(f"resid-{E}x{E}", score(*codec(st)),
                    (E * h + E * (R0 - h)) * (d + nout), 2 * math.ceil(math.log2(E)))
            torch.cuda.empty_cache()
            print(f"    layer {i}: E={E} done ({time.time()-t1:.0f}s)", flush=True)

        # ---- top-2 SUM from one dictionary of rank-h pieces (C(E,2) subspaces) ----
        for E in (a.top2_experts if "top2sum" in a.variants else ()):
            pcs = fit_top2(X, T, E, h, Gh, Ghi, Mf, a.iters)
            add(f"top2sum-{E}", score(lambda Xt, Tt, p=pcs: encode_top2(Xt, Tt, p, Mf),
                                      lambda c, ids, p=pcs: decode_top2(
                                          c, ids, [u for _, u in p])),
                E * h * (d + nout), math.ceil(math.log2(E * (E - 1) // 2)))
            del pcs
            torch.cuda.empty_cache()
            print(f"    layer {i}: top2sum E={E} done ({time.time()-t1:.0f}s)", flush=True)

        # ---- price in rank: equivalent plain rank for each variant's Fisher error ----
        def equiv_rank(err, k):
            rs, es = row["sweep"]["ranks"], row["sweep"]["err"][k]
            for (ra, ea), (rb, eb) in zip(zip(rs, es), zip(rs[1:], es[1:])):
                if ea >= err >= eb:
                    t = (math.log(ea) - math.log(err)) / (math.log(ea) - math.log(eb))
                    return ra * (rb / ra) ** t
            return float("nan") if err < es[-1] else rs[0] * (es[0] / err)
        pl = row["variants"]["plain"]
        print(f"\nlayer {i}  r0={R0}  ({time.time()-t1:.0f}s)   held-out, Fisher metric", flush=True)
        print(f"  {'variant':<17} {'fis fw2':>8} {'dCE fw2':>9} {'x plain':>8} "
              f"{'equiv r':>8} {'cache@eq':>9} {'params':>8} {'bits':>5}", flush=True)
        for name, v in row["variants"].items():
            sc = v["score"]["fineweb2"]
            er = equiv_rank(sc["fis"], "fineweb2")
            v["equiv_rank"] = {k: equiv_rank(v["score"][k]["fis"], k) for k in tests}
            print(f"  {name:<17} {sc['fis']:8.4f} {1e3*sc['dCE']:9.3f} "
                  f"{sc['dCE']/pl['score']['fineweb2']['dCE']:8.3f} {er:8.1f} "
                  f"{100*(R0/er-1) if er == er else float('nan'):+8.1f}% "
                  f"{v['params']/1e6:7.2f}M {v['bits']:5d}", flush=True)
        res[i] = row
        del X, T, tests
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
