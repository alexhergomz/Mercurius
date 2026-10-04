"""Does tied MoL buy cache at FIXED quality? Offline, on real attention inputs, HELD OUT.

WHY (user, 2026-09-28): before any training, take real KV and fit the best latent
projections -- does a mixture reach plain MLA's error at a SMALLER cache?

#43 did a version of this but (a) keys only, (b) IN-SAMPLE -- per-cluster covariances
from ~2x d_model vectors, scored on those same vectors, which flatters MoL -- and the
tied-init lines in train.py are in-sample too. Here:

  FIT   random 1024-token windows from the FIRST half of fineweb_edu_long (stride 2)
  TEST  disjoint windows from its SECOND half, and WikiText (the ppl eval set)
  MODEL the trainer's pre-MLA student: stage A+B, decay seed, native partial RoPE (c0)
  METRIC relative error of the BALANCED joint [K;V] (k_scale/v_scale as LatentKV) --
         the objective the whitened SVD minimises. Pre-k_norm, so a proxy for logits.

  plain   one whitened SVD under the global covariance (= our MLA init, exact)
  tied    spherical k-means on x -> per-cluster whitened SVD; route by cosine to the
          centroids (= mol_tied_cluster_init; the shipped router at step 0)
  best    same pairs refined as k-subspaces: assign each token to the pair that
          reconstructs it best, refit, repeat. At test the token also takes its
          best pair -- DEPLOYABLE (encode with all E, keep the best; E x prefill
          projection cost, zero extra cache), and an upper bound on routing.

HEADLINE: per E, the rank fraction f at which MoL matches plain's held-out error at
the reference allocation r0 (cache/mla_groups_ungrouped_2048.json, 87.5%), i.e.
cache at equal quality = f * 2046 values + ceil(log2 E) bits per token per layer.

    .venv/bin/python experiments/mol_offline_fit.py
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


def windows(tok, path, n, lo, hi, seed, chars=8000, seq=1024):
    """n token windows starting at random CHARACTER offsets in [lo, hi) of the file."""
    g = torch.Generator().manual_seed(seed)
    out = []
    with open(path, "rb") as fh:
        for _ in range(n):
            fh.seek(int(torch.randint(lo, hi - chars, (1,), generator=g)))
            txt = fh.read(chars).decode("utf-8", errors="replace")
            ids = tok(txt, add_special_tokens=False).input_ids[1:seq + 1]
            if len(ids) == seq:
                out.append(torch.tensor(ids))
    return out


@torch.no_grad()
def capture(m, wins, idx, stride=2):
    trunk = get_trunk(m)
    cap = {i: [] for i in idx}

    def mk(i):
        def h(mod, inp, out):
            cap[i].append(inp[0].detach().reshape(-1, inp[0].shape[-1])[::stride]
                          .to("cpu", torch.float16))
        return h
    hs = [trunk.layers[i].self_attn.k_proj.register_forward_hook(mk(i)) for i in idx]
    for w in wins:
        m(input_ids=w.unsqueeze(0).cuda(), logits_to_keep=1)
    for h in hs:
        h.remove()
    return {i: torch.cat(v) for i, v in cap.items()}


def factor(W, cov, rmax):
    """Whitened SVD of W under cov; returns a function r -> (down (r,d), up (out,r))."""
    # fp32 for the factorisation: 8x faster than fp64 on this GPU (2.0 s vs 17.2 s
    # per Cholesky+SVD at 2048x2560), ~120 of them per layer. cov is ACCUMULATED in
    # fp64 (cov_of); fp32 error here is far below the 1e-2 differences measured.
    cov, W = cov.float(), W.float()
    L = whiten_factor(cov)
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    Linv = torch.linalg.solve_triangular(
        L, torch.eye(L.shape[0], device=L.device, dtype=L.dtype), upper=False)
    D = (Vh[:rmax] @ Linv).float()
    Up = (U[:, :rmax] * S[:rmax]).float()
    return lambda r: (D[:r], Up[:, :r])


def cov_of(X):
    X = X.double()
    return (X.t() @ X) / X.shape[0]


def rel_err(X, T, down, up):
    return float((X @ down.t() @ up.t() - T).norm() / T.norm())


def per_token_err(X, T, down, up, bs=8192):
    out = []
    for s in range(0, X.shape[0], bs):
        x = X[s:s + bs]
        out.append(((x @ down.t() @ up.t() - T[s:s + bs]) ** 2).sum(1))
    return torch.cat(out)


def mix_err(X, T, a, facs, r):
    num = 0.0
    for e, f in enumerate(facs):
        m = a == e
        if m.any():
            d, u = f(r)
            num += float(((X[m] @ d.t() @ u.t() - T[m]) ** 2).sum())
    return math.sqrt(num) / float(T.norm())


def best_assign(X, T, facs, r):
    errs = torch.stack([per_token_err(X, T, *f(r)) for f in facs], 1)
    return errs.argmin(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.4, 0.5, 0.625, 0.75, 0.875, 1.0])
    ap.add_argument("--fit-windows", type=int, default=256)     # 131k vectors
    ap.add_argument("--test-windows", type=int, default=64)     # 32k vectors per set
    ap.add_argument("--groups", default="cache/mla_groups_ungrouped_2048.json")
    ap.add_argument("--ksub-iters", type=int, default=3)
    ap.add_argument("--out", default="logs/mol_offline_fit.json")
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
    Xfit = capture(m, windows(tok, FINEWEB_LONG, a.fit_windows, 0, size // 2, 1), idx)
    Xte = {"fineweb2": capture(m, windows(tok, FINEWEB_LONG, a.test_windows,
                                          size // 2, size, 2), idx),
           "wikitext": capture(m, windows(tok, WIKITEXT, a.test_windows, 0,
                                          WIKITEXT.stat().st_size, 3), idx)}
    print(f"captured {Xfit[idx[0]].shape[0]:,} fit / "
          f"{Xte['fineweb2'][idx[0]].shape[0]:,} + {Xte['wikitext'][idx[0]].shape[0]:,}"
          f" test vectors per layer in {time.time()-t0:.0f}s", flush=True)

    res = {}
    for i in idx:
        sa = trunk.layers[i].self_attn
        Wk = sa.k_proj.weight.detach().double()
        Wv = sa.v_proj.weight.detach().double()
        sk = Wk.norm() / math.sqrt(Wk.numel())
        sv = Wv.norm() / math.sqrt(Wv.numel())
        g_ = (sk * sv).sqrt()
        W = torch.cat([Wk * (g_ / sk), Wv * (g_ / sv)])          # balanced, as LatentKV
        Wf = W.float()
        X = Xfit[i].cuda().float()
        T = X @ Wf.t()
        ranks = sorted({max(1, round(f * r0[i])) for f in a.fracs})
        rmax = max(ranks)
        tests = {k: (v[i].cuda().float(), v[i].cuda().float() @ Wf.t())
                 for k, v in Xte.items()}
        row = {"r0": r0[i], "ranks": ranks, "plain": {}, "tied": {}, "best": {}}

        pf = factor(W, cov_of(X), rmax)
        for k, (Xt, Tt) in tests.items():
            row["plain"][k] = [rel_err(Xt, Tt, *pf(r)) for r in ranks]
        row["plain"]["fit"] = [rel_err(X, T, *pf(r)) for r in ranks]

        for E in a.experts:
            C, af = _spherical_kmeans(X, E)
            facs = []
            for e in range(E):
                me = af == e
                facs.append(factor(W, cov_of(X[me]), rmax) if int(me.sum()) >= X.shape[1]
                            else pf)                             # thin cluster: plain
            tr = {"fit": [mix_err(X, T, af, facs, r) for r in ranks]}
            for k, (Xt, Tt) in tests.items():
                at = (F.normalize(Xt, dim=-1) @ C.t()).argmax(1)
                tr[k] = [mix_err(Xt, Tt, at, facs, r) for r in ranks]
            row["tied"][E] = tr

            # k-subspaces refinement at the REFERENCE rank r0, then evaluate all ranks
            ab = af
            for _ in range(a.ksub_iters):
                ab = best_assign(X, T, facs, r0[i])
                facs = [factor(W, cov_of(X[ab == e]), rmax)
                        if int((ab == e).sum()) >= X.shape[1] else pf for e in range(E)]
            br = {"fit": [mix_err(X, T, best_assign(X, T, facs, r), facs, r)
                          for r in ranks]}
            for k, (Xt, Tt) in tests.items():
                br[k] = [mix_err(Xt, Tt, best_assign(Xt, Tt, facs, r), facs, r)
                         for r in ranks]
            br["sizes"] = torch.bincount(ab, minlength=E).tolist()
            row["best"][E] = br
            del facs
            torch.cuda.empty_cache()

        res[i] = row
        j = ranks.index(r0[i]) if r0[i] in ranks else -1
        print(f"\nlayer {i}  r0={r0[i]}  ranks {ranks}", flush=True)
        for k in ["fit", "fineweb2", "wikitext"]:
            print(f"  [{k:>8}] plain {' '.join(f'{v:.4f}' for v in row['plain'][k])}")
            for E in a.experts:
                for kind in ("tied", "best"):
                    v = row[kind][E][k]
                    print(f"  [{k:>8}] {kind} E={E:<2} {' '.join(f'{x:.4f}' for x in v)}"
                          f"   @r0 {100*(v[j]/row['plain'][k][j]-1):+.1f}%", flush=True)
        del X, T, tests
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1)

    # ---- headline: cache needed to match plain@r0 on held-out data ----
    def match_frac(errs, ranks, target, r_ref):
        """Smallest rank reaching `target`, log-linear interpolation; None if never."""
        for (ra, ea), (rb, eb) in zip(zip(ranks, errs), zip(ranks[1:], errs[1:])):
            if ea <= target:
                return ra / r_ref
            if eb <= target:
                t = (math.log(ea) - math.log(target)) / (math.log(ea) - math.log(eb))
                return (ra + t * (rb - ra)) / r_ref
        return None

    print("\n=== cache at EQUAL held-out quality (plain @ r0 = 2046 values/token) ===")
    tot0 = sum(r0.values())
    full = sum(2 * trunk.layers[i].self_attn.k_proj.weight.shape[0] for i in idx)
    summ = {}
    for k in ["fineweb2", "wikitext"]:
        for kind in ("tied", "best"):
            for E in a.experts:
                need, ok = 0.0, True
                for i in idx:
                    row = res[i]
                    j = row["ranks"].index(row["r0"])
                    f = match_frac(row[kind][E][k], row["ranks"],
                                   row["plain"][k][j], row["r0"])
                    if f is None:
                        ok = False
                        f = 1.0
                    need += f * row["r0"]
                bits = math.ceil(math.log2(E)) * len(idx)
                tot = need + bits / 16.0
                tag = "" if ok else "  (some layer never matched in the sweep; counted at r0)"
                summ[f"{k}/{kind}/E{E}"] = tot
                print(f"  [{k:>8}] {kind} E={E:<2}: {tot:7.1f} values/token "
                      f"({100*(1-tot/tot0):+.1f}% cache vs plain; "
                      f"{100*(1 - tot/full):.1f}% total compression){tag}", flush=True)
    json.dump({"layers": res, "summary": summ}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
