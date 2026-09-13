"""Which adapter family can express the update a dense run actually made?

We have the dense solution: val-full trained linear_attn.* densely, so
dW = W_trained - W_base is the update that demonstrably works. Rather than
training four more arms, fit each adapter's parameterization to that dW at
MATCHED parameter budget and measure captured variance.

The families differ along two axes, and this isolates them:

    family   basis            learned            rank        params
    LoRA     adaptive         A and B            <= r        r(m+n)
    VeRA     fixed random     two diagonals      <= r        m + r
    MoRA     fixed random     full square M      <= r_hat    r_hat^2
    Monarch  fixed perms      two block-diags    <= N*r_blk  2*r*(n/N)

LoRA buys an adaptive subspace at low rank. VeRA and MoRA buy high rank inside
a FIXED subspace. Monarch buys cheaper rank. Our measurement says the update is
stable (top-direction overlap 0.994-1.000 between checkpoints) and high rank
(~300 of 1024), which is exactly the regime where that trade should decide it.

Fits:
  LoRA    truncated SVD -- closed form, and an upper bound no rank-r method beats
  VeRA    alternating least squares on the two diagonals (bilinear, each half
          is a diagonal-weighted least squares with a closed-form solution)
  MoRA    closed form: argmin ||D - Dop M Cop||_F is a two-sided pseudoinverse
  Monarch per-block least squares after the Monarch reshape

CAVEAT, stated up front: capturing the dense solution is a NECESSARY condition,
not a sufficient one. A family that cannot express dW certainly cannot reproduce
it; a family that can might still fail to find it by gradient descent, and one
that cannot might still reach comparable loss by a different route. This ranks
plausibility, it does not substitute for training.

MoRA's operators here are fixed RANDOM projections, not the paper's specific
reshape/truncation operators. That keeps the comparison to VeRA honest (both get
a fixed random basis, differing only in what is learned on top) but it is a
stand-in, and a structured operator could do better or worse.
"""
import argparse
import torch
from safetensors import safe_open

from mercurius.paths import CKPT_DIR, STAGE_AB


def load_delta(layer, name, adapters):
    """W_trained - W_base for one matrix, in fp64.

    fp32 svdvals returns NaN on some of these blocks on this machine -- silently,
    producing a full table of NaN. Everything here stays in fp64.
    """
    sd = torch.load(adapters, map_location="cpu")
    key = f"model.layers.{layer}.{name}"
    for cand in (key + ".base.base.weight", key + ".weight", key + ".base.weight"):
        if cand in sd:
            trained = sd[cand].float()
            break
    else:
        return None
    with safe_open(str(STAGE_AB / "model.safetensors"), framework="pt") as fh:
        b = key.replace("model.layers.", "model.language_model.layers.") + ".weight"
        if b not in set(fh.keys()):
            return None
        return (trained - fh.get_tensor(b).float()).double()


def var_captured(D, approx):
    return 1.0 - float((D - approx).pow(2).sum() / D.pow(2).sum())


def fit_lora(D, r):
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def fit_vera(D, r, iters=25, seed=0):
    """D ~ diag(b) B diag(d) A, with A and B frozen random. Learn b and d."""
    g = torch.Generator().manual_seed(seed)
    m, n = D.shape
    A = torch.randn(r, n, generator=g, dtype=torch.float64) / n ** 0.5
    B = torch.randn(m, r, generator=g, dtype=torch.float64) / r ** 0.5
    b = torch.ones(m, dtype=torch.float64)
    d = torch.ones(r, dtype=torch.float64)
    for _ in range(iters):
        # d | b : columns of the model are B[:,k]*b outer A[k,:]; solve the
        # r-dim least squares exactly via the normal equations
        Bb = B * b.unsqueeze(1)                       # (m, r)
        G = (Bb.T @ Bb) * (A @ A.T)                   # (r, r) Gram
        rhs = ((Bb.T @ D) * A).sum(1)                 # (r,)
        d = torch.linalg.solve(G + 1e-10 * torch.eye(r, dtype=torch.float64), rhs)
        # b | d : each row is independent -- a 1-D least squares per row
        M = (B * d.unsqueeze(0)) @ A                  # (m, n)
        num = (M * D).sum(1)
        den = (M * M).sum(1).clamp_min(1e-30)
        b = num / den
    return (B * d.unsqueeze(0) * b.unsqueeze(1)) @ A


def fit_mora(D, rhat, seed=0):
    """D ~ Dop M Cop with Dop, Cop frozen random. Learn the square M.

    argmin over M is closed form: M = Dop^+ D Cop^+.
    """
    g = torch.Generator().manual_seed(seed)
    m, n = D.shape
    Cop = torch.randn(rhat, n, generator=g, dtype=torch.float64) / n ** 0.5
    Dop = torch.randn(m, rhat, generator=g, dtype=torch.float64) / rhat ** 0.5
    M = torch.linalg.pinv(Dop) @ D @ torch.linalg.pinv(Cop)
    return Dop @ M @ Cop


def fit_monarch(D, N, rblk, seed=0):
    """BLOCK-STRUCTURED LOW-RANK BOUND -- not a faithful Monarch fit.

    Read this column as a bracket, not a measurement of MoRE.

    What it computes: D split into N x N blocks, each replaced by its best
    rank-rblk approximation. Those N^2 blocks are INDEPENDENT, whereas a real
    Monarch product P1 L P2 R has only 2N block-diagonal factors and couples
    every block through them. So per unit of rank this is strictly MORE
    expressive than Monarch -- an upper bound on the structure.

    But it is also more EXPENSIVE per unit of rank: N^2 * rblk * (m/N + n/N)
    against Monarch's 2 * r * n/N. At a matched parameter budget on our
    out_proj, real MoRE reaches rank ~976 where this reaches ~324. So budgeting
    honestly for what this fit costs under-ranks Monarch by ~3x, and budgeting
    for Monarch's cost over-states what this fit can do.

    The two errors point in opposite directions and neither is small, which is
    why the result brackets MoRE rather than measuring it. Settling it needs an
    ALS fit of the actual P1 L P2 R product with block-diagonal factors.
    """
    m, n = D.shape
    bm, bn = m // N, n // N
    out = torch.zeros_like(D)
    for i in range(N):
        for j in range(N):
            blk = D[i * bm:(i + 1) * bm, j * bn:(j + 1) * bn]
            U, S, Vh = torch.linalg.svd(blk, full_matrices=False)
            k = min(rblk, S.numel())
            out[i * bm:(i + 1) * bm, j * bn:(j + 1) * bn] = (U[:, :k] * S[:k]) @ Vh[:k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default=str(CKPT_DIR / "adapters-val-full.pt"))
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--matrices", nargs="+",
                    default=["linear_attn.in_proj_qkv", "linear_attn.out_proj"])
    a = ap.parse_args()

    for name in a.matrices:
        D = load_delta(a.layer, name, a.adapters)
        if D is None:
            print(f"{name}: not found in {a.adapters}"); continue
        m, n = D.shape
        S = torch.linalg.svdvals(D)
        c = S.pow(2).cumsum(0) / S.pow(2).sum()
        print(f"\n=== {name}  {tuple(D.shape)}  "
              f"rank@50%={int((c<.5).sum())+1}  dense={m*n/1e6:.2f} M ===")
        print(f"{'budget':>9}  {'LoRA':>16}  {'VeRA':>16}  {'MoRA':>16}  {'Monarch':>16}")
        print(f"{'':9}  {'r / var':>16}  {'r / var':>16}  {'rhat / var':>16}  {'rblk / var':>16}")
        print('-' * 82)
        for P in (0.05e6, 0.11e6, 0.5e6, 1.0e6):
            r_lora = max(1, int(P // (m + n)))
            r_vera = max(1, int(P - m))              # params = m + r
            r_vera = min(r_vera, min(m, n))
            rhat = max(1, int(P ** 0.5))
            rhat = min(rhat, min(m, n))
            N = 4
            # Budget for what fit_monarch ACTUALLY fits: N^2 independent blocks
            # of size (m/N, n/N) at rank rblk. That is an upper bound on true
            # Monarch (which couples the blocks through two shared factors and
            # has only 2N of them), so read this column as a ceiling for
            # block-structured low-rank, not as MoRE's number.
            rblk = max(1, int(P // (N * N * (m // N + n // N))))
            vl = var_captured(D, fit_lora(D, r_lora))
            vv = var_captured(D, fit_vera(D, min(r_vera, min(m, n))))
            vm = var_captured(D, fit_mora(D, rhat))
            vn = var_captured(D, fit_monarch(D, N, rblk))
            print(f"{P/1e6:>8.2f}M  {r_lora:>6} /{vl*100:>7.1f}%  "
                  f"{min(r_vera,512):>6} /{vv*100:>7.1f}%  "
                  f"{rhat:>6} /{vm*100:>7.1f}%  {rblk:>6} /{vn*100:>7.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
