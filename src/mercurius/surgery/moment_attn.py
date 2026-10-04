"""Switch attention's regression type: Nadaraya-Watson -> local linear.

Attention ALREADY IS online kernel regression. out = sum_j w_j v_j with
w_j = softmax(q.k_j/sqrt(d)) is Nadaraya-Watson: a kernel-weighted CONSTANT fit,
solved in closed form at every step from the cached K/V. This module swaps the
estimator for the local LINEAR one, which is lower-bias (Fan 1993: NW carries O(h)
boundary bias and is not minimax efficient; local linear is design-adaptive with
O(h^2) bias everywhere) and still reproduces constants, so the effective weights
still sum to one -- they merely become SIGNED.

    beta_hat = argmin_{b0,B} sum_j w_j || v_j - b0 - B (k_j - q) ||^2
    out      = b0                       the fit evaluated at the query

NO NEW PARAMETERS. This is the point, and I got it wrong once: an earlier version
of this file replaced the solve with learnable directions and per-head gates. That
was backwards -- the closed-form solve is not a defect to be parameterised away,
it is the operator. The only constants here are the projection basis (precomputed
from calibration, like the MLA covariances) and a ridge.

THE MAPPING, with no tricks:
    covariates   X_i        ->  k_j
    responses    Y_i        ->  v_j
    kernel wt  K_h(X_i - x) ->  w_j = softmax(q.k_j/sqrt(d))
    eval point   x          ->  q
    prediction   beta_0     ->  the attention output
The Q/K scale mismatch does NOT break this: B is SOLVED FOR, so it absorbs any
mismatch between the two projections. Nothing needs to be commensurable.

WHY IT IS NOT EXACT AT INSTALL, and why that is fine. This CHANGES THE OPERATOR,
so the pretrained W_Q/W_K/W_V/W_O -- which were trained against an NW aggregator --
are mismatched afterwards, exactly as they are after MLA, GDN-2 or NoPE. The
response is recovery training, not a gate. Offline reconstruction error therefore
does not predict the outcome: the weights adapt. The only test that decides this
is downstream GSM8K after recovery, measured the same way as every other arm.

COST, and no T x T matrix. Every quantity needed is a kernel-weighted sum, i.e. an
attention call with modified values. With the design projected onto m fixed
directions:
    mu_v    = sum_j w_j v_j                      dv channels
    nu_a    = sum_j w_j z_{j,a} v_j              m*dv channels
    m_a     = sum_j w_j z_{j,a}                  m channels
    S_ab    = sum_j w_j z_{j,a} z_{j,b}          m*(m+1)/2 channels
so ONE fused attention call with a widened value tensor gives all of it, and the
per-query work is one (m+1)x(m+1) solve. At m=4 that is a 5x5 solve and a value
width of dv*(m+1) + m + m(m+1)/2.

ABSORPTION SURVIVES, so it ships under MLA:
    z_{j,a} = u_a . (up_k c_j) = (up_k^T u_a) . c_j     a functional of the latent
    nu_a    = up_v ( sum_j w_j z_{j,a} c_j )            m latent accumulators
The KV CACHE IS UNCHANGED -- zero extra bytes per token.
"""
import torch
import torch.nn as nn


class LocalLinearAttn(nn.Module):
    """Parameter-free local-linear aggregation over a fixed direction basis.

    `basis` is (m, d_k), precomputed from calibration and held as a BUFFER, not a
    Parameter: it is a fixed change of coordinates, not something to learn. The
    ridge is a scalar hyperparameter.
    """

    def __init__(self, basis, ridge=1e-3):
        super().__init__()
        self.register_buffer("basis", basis.float(), persistent=True)
        self.m = basis.shape[0]
        self.ridge = float(ridge)

    # ---- the three pieces, separated so each is independently assertable ----

    def z(self, k):
        """z[..., j, a] = u_a . k_j. Feed the LATENT with a pre-folded basis to
        absorb; the algebra is identical."""
        return torch.einsum("...jd,ad->...ja", k.float(), self.basis)

    @staticmethod
    def widen(v, z):
        """[V, z_a V, z_a, z_a z_b] -> one fused attention call gives every moment."""
        m = z.shape[-1]
        nu = (z.unsqueeze(-1) * v.unsqueeze(-2)).reshape(*v.shape[:-1], -1)
        iu = torch.triu_indices(m, m)
        s2 = (z.unsqueeze(-1) * z.unsqueeze(-2))[..., iu[0], iu[1]]
        return torch.cat([v.float(), nu, z, s2], dim=-1)

    def solve(self, out, dv, zq):
        """Turn the widened attention output into beta_0.

        out: (..., dv*(m+1) + m + m(m+1)/2)   zq: (..., m) = basis . q
        Solves the weighted normal equations in the CENTRED coordinates, then
        evaluates at the query, i.e. at delta = zq - (basis . k) => offset -mu_z.
        """
        m = self.m
        mu_v = out[..., :dv]
        nu = out[..., dv:dv + m * dv].reshape(*out.shape[:-1], m, dv)
        mu_z = out[..., dv + m * dv:dv + m * dv + m]
        flat = out[..., dv + m * dv + m:]
        iu = torch.triu_indices(m, m, device=out.device)
        S = out.new_zeros(*out.shape[:-1], m, m)
        S[..., iu[0], iu[1]] = flat
        S = S + S.transpose(-1, -2) - torch.diag_embed(torch.diagonal(S, dim1=-2, dim2=-1))
        # centred second moment and cross moment (weights already sum to 1)
        C = S - mu_z.unsqueeze(-1) * mu_z.unsqueeze(-2)
        R = nu - mu_z.unsqueeze(-1) * mu_v.unsqueeze(-2)
        C = C + self.ridge * torch.eye(m, device=out.device, dtype=C.dtype)
        B = torch.linalg.solve(C, R)                       # (..., m, dv)
        # evaluate the fit at the query: beta_0 = mu_v + (z_q - mu_z)^T B
        return mu_v + torch.einsum("...a,...av->...v", zq - mu_z, B)


def check_against_explicit(seed=0, H=3, T=128, dv=16, m=4, dk=32, ridge=1e-6):
    """The fused path must equal an explicit per-query weighted least squares.

    This is the assertion that matters: #22 and #23 were both mechanisms that
    silently computed something other than what they claimed.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(H, dk, generator=g)
    K = torch.randn(H, T, dk, generator=g)
    V = torch.randn(H, T, dv, generator=g)
    U = torch.linalg.qr(torch.randn(dk, m, generator=g))[0].t()
    mod = LocalLinearAttn(U, ridge=ridge)

    z = mod.z(K)
    w = torch.einsum("hd,htd->ht", q, K).div(dk ** 0.5).softmax(-1)
    wide = mod.widen(V, z)
    fused = mod.solve(torch.einsum("ht,htx->hx", w, wide), dv, mod.z(q.unsqueeze(1))[:, 0])

    # explicit: weighted least squares on [1, z_j - z_q] per head
    explicit = []
    for h in range(H):
        d = z[h] - mod.z(q[h:h + 1].unsqueeze(1))[0, 0]
        X = torch.cat([torch.ones(T, 1), d], 1)
        WX = X * w[h].unsqueeze(1)
        G = X.t() @ WX
        G[1:, 1:] += ridge * torch.eye(m)
        explicit.append(torch.linalg.solve(G, WX.t() @ V[h])[0])
    explicit = torch.stack(explicit)
    err = (fused - explicit).abs().max().item()
    return err, explicit.abs().max().item(), err <= 1e-3 * max(explicit.abs().max().item(), 1)
