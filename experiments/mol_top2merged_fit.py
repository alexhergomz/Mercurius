"""MERGED top-2 MoL (the user's design, #57.1) vs top-1, offline, held out, equal cache/params.

  top-1 E        c = D_e x                       K = U_e c              E maps, rank r
  top-2 merged   c = 1/2 (D_a + D_b) x           K = 1/2 (U_a + U_b) c  E + C(E,2) maps, rank r
                 (averaged, not summed, so a pair of identical experts is exactly one expert;
                  the scale is absorbed by the parameters either way)
Cache: r values + 1 index (top-1) / 2 indices (top-2). Parameters: identical, E*r*(d+out).
Every token's map is rank r, the FULL budget -- nothing is split between experts.

WHY GRADIENT DESCENT. A pair map 1/4 (U_a+U_b)(D_a+D_b) shares parameters bilinearly with
every other pair containing a or b, so the per-cluster SVD that fits top-1 exactly does
not apply. Both models are therefore fitted by the SAME procedure, from the SAME
closed-form top-1 start, for the SAME number of Adam steps -- so any difference is the
structure, not the optimiser:
    rounds x { exact best assignment (top-1: best expert; top-2: best of C(E,2) pairs,
               computed from x and the fixed W, as offline best-of-E always was)
               -> Adam on the relative balanced [K;V] error with assignments fixed }
INIT. Closed-form top-1 pieces (per-cluster whitened SVD, k-subspaces) live in different
latent frames; averaging D_a, D_b across frames is meaningless. Each expert is gauge-
aligned to the plain factorisation's frame by orthogonal Procrustes (D_e <- R_e D_e,
U_e <- U_e R_e^T, R_e = polar(D_0 D_e^T)) -- exact for top-1, which is gauge-invariant.

REPORTED held out (fineweb second half, WikiText): relative balanced [K;V] error for
plain, top-1 (closed form), top-1 (GD), top-2 merged (GD), and the pair-usage spread.

    .venv/bin/python experiments/mol_top2merged_fit.py --layers 3 7 19 31
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import torch

from mercurius.models.kda import load_kda_model
from mercurius.paths import FINEWEB_LONG, STAGE_AB, WIKITEXT
from mercurius.surgery.mol_struct import Metric, rrr, _ksub
from mercurius.surgery.norm_fusion import get_trunk

sys.path.insert(0, os.path.dirname(__file__))
from mol_offline_fit import windows, capture                     # noqa: E402


def procrustes_align(pieces, D0):
    out = []
    for D, U in pieces:
        M = D0 @ D.t()                               # (r, r)
        P, _, Qh = torch.linalg.svd(M)
        R = P @ Qh                                   # polar factor: argmin ||R D - D0||
        out.append((R @ D, U @ R.t()))
    return out


class Model(torch.nn.Module):
    def __init__(self, pieces):
        super().__init__()
        self.D = torch.nn.Parameter(torch.stack([p[0] for p in pieces]).clone())   # (E,r,d)
        self.U = torch.nn.Parameter(torch.stack([p[1] for p in pieces]).clone())   # (E,out,r)
        E = self.D.shape[0]
        self.pairs = torch.tensor(list(itertools.combinations(range(E), 2)),
                                  device=self.D.device)                           # (P,2)

    def maps(self, kind):
        """List of (down, up) per selectable unit: experts (top1) or pairs (top2)."""
        if kind == "top1":
            return [(self.D[e], self.U[e]) for e in range(self.D.shape[0])]
        return [(0.5 * (self.D[a] + self.D[b]), 0.5 * (self.U[a] + self.U[b]))
                for a, b in self.pairs.tolist()]


@torch.no_grad()
def assign(X, T, maps, M, bs=4096):
    out = []
    for s in range(0, X.shape[0], bs):
        x, t = X[s:s + bs], T[s:s + bs]
        out.append(torch.stack([M.sq(x @ d.t() @ u.t() - t) for d, u in maps], 1).argmin(1))
    return torch.cat(out)


def rel_err(X, T, maps, a, M):
    num = 0.0
    for j, (d, u) in enumerate(maps):
        m = a == j
        if m.any():
            num = num + M.sq(X[m] @ d.t() @ u.t() - T[m]).sum()
    return num / M.sq(T).sum()


def fit(model, kind, X, T, M, rounds, steps, lr, bs=16384):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    den = float(M.sq(T).sum()) / X.shape[0]
    for rd in range(rounds):
        with torch.no_grad():
            a = assign(X, T, [(d.detach(), u.detach()) for d, u in model.maps(kind)], M)
        for st in range(steps):
            idx = torch.randint(0, X.shape[0], (bs,), device=X.device)
            x, t, aa = X[idx], T[idx], a[idx]
            maps = model.maps(kind)
            loss = 0.0
            for j, (d, u) in enumerate(maps):
                m = aa == j
                if m.any():
                    loss = loss + M.sq(x[m] @ d.t() @ u.t() - t[m]).sum()
            loss = loss / (bs * den)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            full = float(rel_err(X, T, [(d.detach(), u.detach()) for d, u in model.maps(kind)],
                                 a, M)) ** 0.5
        print(f"      {kind} round {rd}: train rel err {full:.4f}", flush=True)
    with torch.no_grad():
        return assign(X, T, [(d.detach(), u.detach()) for d, u in model.maps(kind)], M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 7, 19, 31])
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--groups", default="cache/mla_groups_ungrouped_2048.json")
    ap.add_argument("--fit-windows", type=int, default=256)
    ap.add_argument("--test-windows", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="logs/mol_top2merged_fit.json")
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
    Xfit = capture(m, windows(tok, FINEWEB_LONG, a.fit_windows, 0, size // 2, 1), idx)
    Xte = {"fineweb2": capture(m, windows(tok, FINEWEB_LONG, a.test_windows, size // 2, size, 2), idx),
           "wikitext": capture(m, windows(tok, WIKITEXT, a.test_windows, 0,
                                          WIKITEXT.stat().st_size, 3), idx)}
    print(f"captured {Xfit[idx[0]].shape[0]:,} fit vectors per layer", flush=True)

    res = {}
    for i in idx:
        t0 = time.time()
        sa = trunk.layers[i].self_attn
        Wk, Wv = sa.k_proj.weight.detach().float(), sa.v_proj.weight.detach().float()
        W = torch.cat([Wk, Wv])
        sk = Wk.norm() / math.sqrt(Wk.numel()); sv = Wv.norm() / math.sqrt(Wv.numel())
        g_ = (sk * sv).sqrt()
        sc = torch.cat([torch.full((Wk.shape[0],), float(g_ / sk)),
                        torch.full((Wv.shape[0],), float(g_ / sv))]).cuda()
        A, Ai = torch.diag(sc), torch.diag(1 / sc)
        M = Metric(A)
        # unit-RMS inputs and targets: relative errors are invariant, and it keeps the
        # whitened factors O(1) so one Adam lr is sane for every layer
        X = Xfit[i].cuda().float()
        sx = X.pow(2).mean().sqrt(); X = X / sx
        W = W / (X @ W.t()).pow(2).mean().sqrt()
        T = X @ W.t()
        r = r0[i]
        tests = {k: (v[i].cuda().float() / sx, (v[i].cuda().float() / sx) @ W.t())
                 for k, v in Xte.items()}

        D0, U0 = rrr(X, T, A, Ai, r)
        row = {"r0": r}

        def score(maps, name):
            row[name] = {}
            for k, (Xt, Tt) in tests.items():
                at = assign(Xt, Tt, maps, M)
                row[name][k] = float(rel_err(Xt, Tt, maps, at, M)) ** 0.5
                if name.startswith("top2"):
                    row[name][k + "_pairs_used"] = int(torch.unique(at).numel())
        score([(D0, U0)], "plain")
        pieces = _ksub(X, T, a.E, r, A, Ai, M, 2)
        score(pieces, "top1_closed")
        aligned = procrustes_align(pieces, D0)
        score(aligned, "top1_aligned")          # sanity: must equal top1_closed
        m1 = Model(aligned); fit(m1, "top1", X, T, M, a.rounds, a.steps, a.lr)
        score([(d.detach(), u.detach()) for d, u in m1.maps("top1")], "top1_gd")
        m2 = Model(aligned); fit(m2, "top2", X, T, M, a.rounds, a.steps, a.lr)
        score([(d.detach(), u.detach()) for d, u in m2.maps("top2")], "top2m_gd")
        res[i] = row
        print(f"\nlayer {i} r={r} ({time.time()-t0:.0f}s)  held-out relative balanced [K;V] error",
              flush=True)
        for k in tests:
            p = row["plain"][k]
            line = "  ".join(f"{n} {row[n][k]:.4f} ({100*(row[n][k]/p-1):+.1f}%)"
                             for n in ("plain", "top1_closed", "top1_aligned", "top1_gd", "top2m_gd"))
            print(f"  [{k:>8}] {line}  pairs used {row['top2m_gd'][k + '_pairs_used']}/"
                  f"{a.E*(a.E-1)//2}", flush=True)
        del X, T, tests, m1, m2
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
