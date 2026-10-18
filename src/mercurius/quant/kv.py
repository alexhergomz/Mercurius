"""KV-cache quantizers for quantization-aware training: the forward sees exactly the cached values the deployed
runtime will read back, the backward is straight-through.

Each quantizer maps x (B, T, C) -> x_hat (B, T, C) along the token axis T (the FoveaKV block means need it) and is
applied to whatever the attention layer caches: the MLA latent c and the decoupled RoPE key (absorbable MLA), or
the per-head keys (post-RoPE) and values of a plain attention layer.

    foveakv     FoveaKV's gain-shape format (github.com/Minerva-Laboratories/FoveaKV, GainShape mode "gs"): per
                head of `head_dim` channels, subtract the mean of the token's 128-token block (fp16), rotate with a
                fixed randomized Hadamard transform, store the norm as an 8-bit log-uniform gain (range per head over
                the sequence) and the unit shape coordinate-wise on the 16-level Gaussian Lloyd-Max codebook, nested
                as a 2-bit coarse plane + 2-bit residual plane. bits=4 reads both planes (what the decode kernels read
                for every opened page), bits=2 the coarse plane only. Shrinkage corrections (keys renormalized,
                values unbiased) are FoveaKV's optional decode-side debiasing; off by default for a shared latent.
    turboquant  TurboQuant-MSE without QJL (Mercurius-1): fixed Haar rotation, fp16 norm per token, Beta Lloyd-Max
                codebook for a random unit vector's coordinates.
    int         symmetric int, fp16 scale per `group` channels.
    none        identity.

FoveaKV keeps the most recent 128..255 tokens exact (bf16 window); `window` reproduces that per sequence end only
(quantization is per cached token, not per query), so the default is to quantize every token -- slightly
pessimistic, never optimistic.
"""
from __future__ import annotations

import math

import torch

_CODEBOOKS = {}


def _phi(x):
    return torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def _cond_mean(a, b):
    """E[X | a < X < b], X ~ N(0, 1)."""
    return (_phi(a) - _phi(b)) / (_Phi(b) - _Phi(a)).clamp_min(1e-300)


def gaussian_lloyd_max(n, iters=500):
    """(boundaries, levels) of the n-level Lloyd-Max quantizer for N(0, 1), fp64."""
    lev = torch.linspace(-2.5, 2.5, n, dtype=torch.float64)
    inf = torch.tensor([float("inf")], dtype=torch.float64)
    for _ in range(iters):
        bnd = (lev[1:] + lev[:-1]) / 2
        e = torch.cat([-inf, bnd, inf])
        lev = _cond_mean(e[:-1], e[1:])
    return (lev[1:] + lev[:-1]) / 2, lev


def nested_codebook():
    """FoveaKV's NestedCodebook(4, 2): fine 16-level Lloyd-Max; coarse level c = conditional mean over the 4 fine
    cells sharing its 2 high bits. Returns fp32 (bnd16, lut16, lut4, mse_fine, mse_coarse)."""
    if "nested" not in _CODEBOOKS:
        bnd, lut16 = gaussian_lloyd_max(16)
        inf = torch.tensor([float("inf")], dtype=torch.float64)
        e = torch.cat([-inf, bnd, inf])
        ce = e[::4]
        if ce[-1] != float("inf"):
            ce = torch.cat([ce, inf])
        lut4 = _cond_mean(ce[:-1], ce[1:])

        def mse(b, lv):
            ee = torch.cat([-inf, b, inf])
            tot = 0.0
            for i, c in enumerate(lv):
                a, bb = ee[i], ee[i + 1]
                p = _Phi(bb) - _Phi(a)
                m2 = p - (bb * _phi(bb) if torch.isfinite(bb) else 0.0) + (a * _phi(a) if torch.isfinite(a) else 0.0)
                tot += float(m2 - 2 * c * (_phi(a) - _phi(bb)) + c * c * p)
            return tot
        _CODEBOOKS["nested"] = (bnd.float(), lut16.float(), lut4.float(), mse(bnd, lut16), mse(ce[1:-1], lut4))
    return _CODEBOOKS["nested"]


def hadamard(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    if H.shape[0] != n:
        raise ValueError(f"randomized Hadamard needs a power-of-two head width, got {n}")
    return H


def rht(n, seed):
    """Orthonormal randomized Hadamard H diag(s) / sqrt(n) (FoveaKV's rotation, normalized)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
    return (hadamard(n) * s.unsqueeze(0) / math.sqrt(n)).float()


def _ste(x, xq):
    return x + (xq - x).detach()


class NoKV:
    def __call__(self, x, kind="kv"):
        return x


class IntKV:
    def __init__(self, bits=4, group=32):
        self.bits, self.group = bits, group

    def __call__(self, x, kind="kv"):
        qmax = 2 ** (self.bits - 1) - 1
        C = x.shape[-1]
        pad = (-C) % self.group
        xf = torch.nn.functional.pad(x.float(), (0, pad)) if pad else x.float()
        g = xf.reshape(*xf.shape[:-1], -1, self.group)
        s = (g.abs().amax(-1, keepdim=True) / qmax).half().float().clamp_min(1e-8)
        q = ((g / s).round().clamp(-qmax, qmax) * s).reshape(xf.shape)[..., :C]
        return _ste(x, q.to(x.dtype))


class TurboQuantKV:
    """Haar rotation per width (seeded), fp16 norm, Beta(r) Lloyd-Max codebook (Mercurius-1's cache)."""

    def __init__(self, bits=4, seed=0):
        self.bits, self.seed, self.rot, self.books = bits, seed, {}, {}

    def _book(self, r):
        if r not in self.books:
            x = torch.linspace(-1, 1, 400001, dtype=torch.float64)[1:-1]
            w = (1 - x * x).clamp_min(0) ** ((r - 3) / 2)
            w = w / w.sum()
            cdf = w.cumsum(0)
            L = 2 ** self.bits
            c = x[torch.searchsorted(cdf, (torch.arange(L, dtype=torch.float64) + 0.5) / L)]
            for _ in range(300):
                b = (c[1:] + c[:-1]) / 2
                idx = torch.bucketize(x, b)
                num = torch.zeros(L, dtype=torch.float64).index_add_(0, idx, w * x)
                den = torch.zeros(L, dtype=torch.float64).index_add_(0, idx, w)
                c = torch.where(den > 0, num / den.clamp_min(1e-300), c)
            self.books[r] = (c.float(), ((c[1:] + c[:-1]) / 2).float())
        return self.books[r]

    def _R(self, r, dev):
        if r not in self.rot:
            g = torch.Generator().manual_seed(self.seed * 1000 + r)
            Q, Rr = torch.linalg.qr(torch.randn(r, r, generator=g, dtype=torch.float64))
            self.rot[r] = (Q * torch.sign(torch.diagonal(Rr))).float()
        return self.rot[r].to(dev)

    def __call__(self, x, kind="kv"):
        r = x.shape[-1]
        R = self._R(r, x.device)
        cen, bnd = (t.to(x.device) for t in self._book(r))
        y = x.float() @ R.T
        n = y.norm(dim=-1, keepdim=True).half().float().clamp_min(1e-12)
        q = cen[torch.bucketize(y / n, bnd)] * n
        return _ste(x, (q @ R).to(x.dtype))


class FoveaKV:
    """FoveaKV gain-shape quantization of x (B, T, C), C split into C / head_dim heads."""

    def __init__(self, bits=4, head_dim=128, block=128, gain_bits=8, shrink=False, window=0, seed=0):
        if bits not in (2, 4):
            raise ValueError("FoveaKV planes: bits 4 (coarse + residual) or 2 (coarse only)")
        self.bits, self.hd, self.block, self.gain_bits = bits, head_dim, block, gain_bits
        self.shrink, self.window, self.seed = shrink, window, seed
        self._R = {}

    def _rot(self, n, dev):
        if n not in self._R:
            self._R[n] = rht(n, self.seed)
        return self._R[n].to(dev)

    def _block_means(self, x):
        """Mean of each token's `block`-token block (the trailing partial block uses its own mean), fp16."""
        B, T = x.shape[:2]
        nb = T // self.block
        M = torch.empty_like(x)
        if nb:
            full = x[:, :nb * self.block].reshape(B, nb, self.block, *x.shape[2:]).mean(2)
            M[:, :nb * self.block] = full.repeat_interleave(self.block, 1)
        if T % self.block:
            M[:, nb * self.block:] = x[:, nb * self.block:].mean(1, keepdim=True)
        return M.half().float()

    def quantize(self, x, kind="kv"):
        bnd16, lut16, lut4, mse_f, mse_c = (t if isinstance(t, float) else t.to(x.device) for t in nested_codebook())
        B, T, C = x.shape
        hd = self.hd if C % self.hd == 0 else C            # a narrow tensor (e.g. a 64-wide RoPE key) is one head
        xf = x.float().reshape(B, T, C // hd, hd)
        M = self._block_means(xf)
        R = self._rot(hd, x.device)
        y = (xf - M) @ R.T
        g = y.norm(dim=-1)                                                   # (B, T, heads)
        z = y / g.clamp_min(1e-30).unsqueeze(-1) * math.sqrt(hd)
        idx = torch.bucketize(z, bnd16)
        lev = lut16[idx] if self.bits == 4 else lut4[idx >> 2]
        if self.shrink:
            D = mse_f if self.bits == 4 else mse_c
            lev = lev * ((1 - D) ** -0.5 if kind == "k" else 1.0 / (1 - D))
        n = 1 << self.gain_bits
        lg = torch.log(g.clamp_min(1e-30))
        lo = lg.amin(1, keepdim=True).detach()                               # range per (sequence, head)
        hi = lg.amax(1, keepdim=True).detach()
        span = (hi - lo).clamp_min(1e-12)
        c = torch.floor((lg - lo) / span * n).clamp(0, n - 1)
        gq = torch.exp(lo + (c + 0.5) / n * span)
        xq = M + (gq.unsqueeze(-1) * lev / math.sqrt(hd)) @ R
        xq = xq.reshape(B, T, C)
        if self.window and T > self.window:
            keep = T - (T - self.window) // self.block * self.block          # FoveaKV's exact tail: 128..255
            xq = torch.cat([xq[:, :T - keep], x.float()[:, T - keep:]], 1)
        return xq.to(x.dtype)

    def __call__(self, x, kind="kv"):
        with torch.no_grad():
            xq = self.quantize(x, kind)
        return _ste(x, xq)


def make_kv_quantizer(cfg_kv):
    f = cfg_kv["format"]
    if f == "none":
        return NoKV()
    if f == "int":
        return IntKV(cfg_kv["bits"], cfg_kv["group"])
    if f == "turboquant":
        return TurboQuantKV(cfg_kv["bits"], cfg_kv["seed"])
    if f == "foveakv":
        return FoveaKV(cfg_kv["bits"], cfg_kv["head_dim"], cfg_kv["block"], cfg_kv["gain_bits"], cfg_kv["shrink"],
                       cfg_kv.get("window", 0), cfg_kv["seed"])
    raise ValueError(f)
