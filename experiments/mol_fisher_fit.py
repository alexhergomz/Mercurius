"""Tied MoL vs plain MLA at fixed quality, in a LOSS-RELEVANT metric (output Fisher).

WHY (user, 2026-09-28): mol_offline_fit.py measured relative error of the BALANCED
[K;V] activations -- the input-whitened (CARE) objective. That weights every K and V
direction equally and knows nothing of k_norm, RoPE, q.k, the attention weights or
o_proj; the K/V balance is a heuristic. "Plain doesn't really mean anything."

OUTPUT FISHER. Per attention layer capture g_t = dL/d[k_t; v_t] (the k_proj/v_proj
outputs, natural units) on real windows under the LM loss, G = E_t[g_t g_t^T]
(2048x2048). Second order, empirical-Fisher approximation of the Hessian block:
    dL ~= 1/2 * E_t[ d_t^T G d_t ]  per token, d_t the reconstruction error.
k_norm is scale-invariant per head, so g_t is orthogonal to k_t within each head and G
correctly discounts errors along k -- one of the things the balanced metric gets wrong.

FITS (each for plain E=1 and every MoL pair):
  cov     whitened SVD of S W L, S = diag(k_scale, v_scale): the pipeline's CARE fit
  fisher  SVD of G^1/2 W L: exact minimiser of E||G^1/2 (W - W^) x||^2, because the
          weight (input cov (x) output Fisher) is Kronecker-separable (Manton, Mahony &
          Hua 2003 Thm 3). up = G^-1/2 U S, down = V^T L^-1.
METRICS on HELD-OUT inputs (G from the FIT windows only -- no test gradients):
  bal     relative balanced [K;V] error (continuity with mol_offline_fit.py)
  fis     relative Fisher error sqrt(tr(D G D^T) / tr(T G T^T))
  dCE     1/2 mean_t d_t^T G d_t, nats/token: an absolute second-order loss proxy,
          summed over layers comparable to the step-0 ppl smokes (sign/size check)
ROUTERS: cosine to fit-set centroids (the shipped router), and best-of-E under the
Fisher metric (uses only x and fixed G, W: deployable at prefill).

HEADLINE: rank fraction at which each MoL matches its own plain fit's held-out error at
r0, in each fit's own metric (cov/bal, fisher/fis), plus cov-fit judged by Fisher.

    .venv/bin/python experiments/mol_fisher_fit.py
"""
import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

from mercurius.models.kda import load_kda_model
from mercurius.paths import FINEWEB_LONG, STAGE_AB, WIKITEXT
from mercurius.surgery.mol import _spherical_kmeans
from mercurius.surgery.norm_fusion import get_trunk
from mercurius.surgery.transmla import whiten_factor

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mol_offline_fit import windows, capture, cov_of          # noqa: E402


def capture_fisher(m, wins, idx):
    """G_i = mean over tokens of g g^T, g = dL/d[k_proj out; v_proj out], L = summed CE."""
    trunk = get_trunk(m)
    for p in m.parameters():
        p.requires_grad_(False)
    G = {i: None for i in idx}
    n = {i: 0 for i in idx}
    buf = {}

    def fwd(i, which):
        def h(mod, inp, out):
            out.register_hook(lambda g: buf.__setitem__((i, which), g.detach()))
        return h
    hs = []
    for i in idx:
        sa = trunk.layers[i].self_attn
        hs.append(sa.k_proj.register_forward_hook(fwd(i, "k")))
        hs.append(sa.v_proj.register_forward_hook(fwd(i, "v")))
    emb = m.get_input_embeddings()
    hs.append(emb.register_forward_hook(lambda mod, inp, out: out.requires_grad_(True)))
    for s, w in enumerate(wins):
        ids = w.unsqueeze(0).cuda()
        logits = m(input_ids=ids).logits[0, :-1].float()
        loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum")
        loss.backward()
        del logits, loss
        for i in idx:
            g = torch.cat([buf[(i, "k")], buf[(i, "v")]], -1)
            g = g.reshape(-1, g.shape[-1]).double()
            G[i] = g.t() @ g if G[i] is None else G[i] + g.t() @ g
            n[i] += g.shape[0]
        buf.clear()
        if (s + 1) % 64 == 0:
            print(f"    fisher: {s+1}/{len(wins)} windows", flush=True)
    for h in hs:
        h.remove()
    return {i: (G[i] / n[i]) for i in idx}


def sqrt_pair(G, damp):
    """(G + damp*mean_eig*I)^(1/2) and ^(-1/2), fp64 on CPU."""
    w, Q = torch.linalg.eigh(G.cpu())
    w = w.clamp_min(0) + damp * float(w.clamp_min(0).mean())
    return ((Q * w.sqrt()) @ Q.t()).float().cuda(), ((Q * w.rsqrt()) @ Q.t()).float().cuda()


def factor(W, cov, rmax, A, Ai):
    """min E||A (W - W^) x||^2 at every r <= rmax. W natural units; A output transform."""
    L = whiten_factor(cov.float())
    U, S, Vh = torch.linalg.svd(A @ W @ L, full_matrices=False)
    Linv = torch.linalg.solve_triangular(
        L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    D = Vh[:rmax] @ Linv
    Up = Ai @ (U[:, :rmax] * S[:rmax])
    return lambda r: (D[:r], Up[:, :r])


class Metric:
    """Squared error of D = X W^T - T in one output metric, summed over rows."""
    def __init__(self, A):
        self.A = A                                      # rows of D are right-multiplied by A^T

    def sq(self, Dlt):
        return ((Dlt @ self.A.t()) ** 2).sum(1)


def eval_mix(X, T, a, facs, r, metrics, bs=8192):
    """Summed squared error per metric for assignment a (token -> pair)."""
    tot = {k: 0.0 for k in metrics}
    for e, f in enumerate(facs):
        me = (a == e).nonzero().squeeze(1)
        if me.numel() == 0:
            continue
        d, u = f(r)
        for s in range(0, me.numel(), bs):
            j = me[s:s + bs]
            Dlt = X[j] @ d.t() @ u.t() - T[j]
            for k, M in metrics.items():
                tot[k] += float(M.sq(Dlt).sum())
    return tot


def best_assign(X, T, facs, r, M, bs=8192):
    out = []
    for s in range(0, X.shape[0], bs):
        x, t = X[s:s + bs], T[s:s + bs]
        out.append(torch.stack([M.sq(x @ f(r)[0].t() @ f(r)[1].t() - t) for f in facs],
                               1).argmin(1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.4, 0.5, 0.625, 0.75, 0.875, 1.0])
    ap.add_argument("--fit-windows", type=int, default=256)
    ap.add_argument("--fisher-windows", type=int, default=256)
    ap.add_argument("--test-windows", type=int, default=64)
    ap.add_argument("--groups", default="cache/mla_groups_ungrouped_2048.json")
    ap.add_argument("--damp", type=float, default=1e-3)
    ap.add_argument("--ksub-iters", type=int, default=3)
    ap.add_argument("--out", default="logs/mol_fisher_fit.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(STAGE_AB))
    m = load_kda_model(str(STAGE_AB), dtype=torch.bfloat16)
    for l in get_trunk(m).layers:
        if hasattr(l, "linear_attn"):
            l.linear_attn.seed_decay_from_rope(target_alpha=None)
    m.eval()
    trunk = get_trunk(m)
    idx = [i for i, l in enumerate(trunk.layers) if hasattr(l, "self_attn")]
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
    print(f"captured inputs in {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    Gs = capture_fisher(m, fit_w[:a.fisher_windows], idx)
    print(f"output Fisher from {min(a.fisher_windows, len(fit_w))} windows in "
          f"{time.time()-t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

    res = {}
    for i in idx:
        sa = trunk.layers[i].self_attn
        Wk = sa.k_proj.weight.detach().float()
        Wv = sa.v_proj.weight.detach().float()
        W = torch.cat([Wk, Wv])                                   # natural units
        sk = Wk.norm() / math.sqrt(Wk.numel())
        sv = Wv.norm() / math.sqrt(Wv.numel())
        g_ = (sk * sv).sqrt()
        sc = torch.cat([torch.full((Wk.shape[0],), float(g_ / sk)),
                        torch.full((Wv.shape[0],), float(g_ / sv))]).cuda()
        A_bal, Ai_bal = torch.diag(sc), torch.diag(1 / sc)
        Gh, Ghi = sqrt_pair(Gs[i], a.damp)
        metrics = {"bal": Metric(A_bal), "fis": Metric(Gh)}
        rawG = Metric(sqrt_pair(Gs[i], 0.0)[0])          # undamped: dCE = 1/2 d^T G d
        fits = {"cov": (A_bal, Ai_bal), "fisher": (Gh, Ghi)}

        X = Xfit[i].cuda().float()
        T = X @ W.t()
        ranks = sorted({max(1, round(f * r0[i])) for f in a.fracs})
        rmax = max(ranks)
        tests = {k: (v[i].cuda().float(), v[i].cuda().float() @ W.t()) for k, v in Xte.items()}
        den = {k: {mk: float(M.sq(Tt).sum()) for mk, M in metrics.items()}
               for k, (Xt, Tt) in tests.items()}
        row = {"r0": r0[i], "ranks": ranks}
        cov_all = cov_of(X)

        def record(tag, per_set):
            row[tag] = per_set

        for fk, (A, Ai) in fits.items():
            pf = factor(W, cov_all, rmax, A, Ai)
            out = {}
            for k, (Xt, Tt) in tests.items():
                z = torch.zeros(Xt.shape[0], dtype=torch.long, device="cuda")
                errs = [eval_mix(Xt, Tt, z, [pf], r, metrics) for r in ranks]
                out[k] = {mk: [math.sqrt(e[mk] / den[k][mk]) for e in errs]
                          for mk in metrics}
                out[k]["dCE"] = [0.5 * float(rawG.sq(Xt @ pf(r)[0].t() @ pf(r)[1].t() - Tt).mean())
                                 for r in ranks]
            record(f"plain/{fk}", out)

            for E in a.experts:
                C, af = _spherical_kmeans(X, E)
                facs = [factor(W, cov_of(X[af == e]), rmax, A, Ai)
                        if int((af == e).sum()) >= X.shape[1] else pf for e in range(E)]
                out = {}
                for k, (Xt, Tt) in tests.items():
                    at = (F.normalize(Xt, dim=-1) @ C.t()).argmax(1)
                    errs = [eval_mix(Xt, Tt, at, facs, r, metrics) for r in ranks]
                    out[k] = {mk: [math.sqrt(e[mk] / den[k][mk]) for e in errs]
                              for mk in metrics}
                    out[k]["dCE"] = [0.5 * eval_mix(Xt, Tt, at, facs, r, {"g": rawG})["g"]
                                     / Xt.shape[0] for r in ranks]
                record(f"tied{E}/{fk}", out)

                # k-subspaces under THIS fit's own metric, then best-of-E at test
                M = metrics["bal" if fk == "cov" else "fis"]
                ab = af
                for _ in range(a.ksub_iters):
                    ab = best_assign(X, T, facs, r0[i], M)
                    facs = [factor(W, cov_of(X[ab == e]), rmax, A, Ai)
                            if int((ab == e).sum()) >= X.shape[1] else pf for e in range(E)]
                out = {}
                for k, (Xt, Tt) in tests.items():
                    errs = [eval_mix(Xt, Tt, best_assign(Xt, Tt, facs, r, M), facs, r,
                                     metrics) for r in ranks]
                    out[k] = {mk: [math.sqrt(e[mk] / den[k][mk]) for e in errs]
                              for mk in metrics}
                record(f"best{E}/{fk}", out)
                del facs
                torch.cuda.empty_cache()

        res[i] = row
        j = ranks.index(r0[i])
        print(f"\nlayer {i}  r0={r0[i]}  held-out error at r0 "
              f"(bal | fis | dCE nats/tok x1e3)", flush=True)
        for k in tests:
            for tag in [t for t in row if "/" in t]:
                v = row[tag][k]
                dce = f"{1e3*v['dCE'][j]:7.3f}" if "dCE" in v else "      -"
                print(f"  [{k:>8}] {tag:<14} {v['bal'][j]:.4f} | {v['fis'][j]:.4f} | {dce}",
                      flush=True)
        del X, T, tests
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1)

    # ---- headline: cache at equal held-out quality, each fit in its own metric ----
    def match_frac(errs, ranks, target, r_ref):
        for (ra, ea), (rb, eb) in zip(zip(ranks, errs), zip(ranks[1:], errs[1:])):
            if ea <= target:
                return ra / r_ref
            if eb <= target:
                t = (math.log(ea) - math.log(target)) / (math.log(ea) - math.log(eb))
                return (ra + t * (rb - ra)) / r_ref
        return None

    tot0 = sum(r0.values())
    summ = {}
    print(f"\n=== cache at EQUAL held-out quality vs same-fit plain @ r0 "
          f"({tot0} values/token) ===")
    for fk, mk in [("cov", "bal"), ("cov", "fis"), ("fisher", "fis")]:
        for k in ["fineweb2", "wikitext"]:
            for kind in ("tied", "best"):
                for E in a.experts:
                    need, ok = 0.0, True
                    for i in idx:
                        row = res[i]
                        j = row["ranks"].index(row["r0"])
                        f = match_frac(row[f"{kind}{E}/{fk}"][k][mk], row["ranks"],
                                       row[f"plain/{fk}"][k][mk][j], row["r0"])
                        if f is None:
                            ok, f = False, 1.0
                        need += f * row["r0"]
                    tot = need + math.ceil(math.log2(E)) * len(idx) / 16.0
                    summ[f"{fk}-fit/{mk}/{k}/{kind}{E}"] = tot
                    print(f"  fit={fk:<6} metric={mk} [{k:>8}] {kind} E={E:<2}: "
                          f"{tot:7.1f} values/token ({100*(1-tot/tot0):+.1f}% cache)"
                          f"{'' if ok else '  (some layer unmatched, counted at r0)'}",
                          flush=True)
    # plain-to-plain: how much does the Fisher fit alone buy over CARE, in Fisher metric
    for k in ["fineweb2", "wikitext"]:
        need = 0.0
        for i in idx:
            row = res[i]
            j = row["ranks"].index(row["r0"])
            f = match_frac(row["plain/fisher"][k]["fis"], row["ranks"],
                           row["plain/cov"][k]["fis"][j], row["r0"])
            need += (f or 1.0) * row["r0"]
        summ[f"plainfisher_vs_plaincov/{k}"] = need
        print(f"  plain Fisher-fit vs plain CARE-fit, Fisher metric [{k:>8}]: "
              f"{need:7.1f} values/token ({100*(1-need/tot0):+.1f}% cache)", flush=True)
    for k in ["fineweb2", "wikitext"]:
        s = {tag: sum(res[i][tag][k]["dCE"][res[i]["ranks"].index(res[i]["r0"])]
                      for i in idx) for tag in res[idx[0]] if "/" in tag
             and "dCE" in res[idx[0]][tag][k]}
        print(f"  summed dCE proxy @ r0 [{k:>8}]: " +
              "  ".join(f"{t} {v:.4f}" for t, v in s.items()), flush=True)
    json.dump({"layers": res, "summary": summ}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
