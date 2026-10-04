"""Two capacity extensions for the MLA latent, both exactly function-preserving.

The rank ceiling, stated precisely. With K_j = up_k . c_j and c_j in R^r, every key
lies in range(up_k) -- an r-dimensional subspace -- so the attention score is a
bilinear form of rank <= r. Measured 2026-09-26 (decisions #25), that ceiling
BINDS: doubling r kept 73% of its advantage through training, while redistributing
a fixed r kept only 8%. No linear reparameterisation of a single up_k escapes it
(Monarch, butterfly and block-sparse are all restrictions of dense, hence <= rank
r), so the two escapes are more taps or a nonlinearity.

(1) GATED LATENT -- a GLU on the latent, which is the compressed intermediate.

        c      = down(x)
        c_tilde = c * (1 + act(down_g(x)))
        cache c_tilde;  K = up_k . c_tilde

    Gating an intermediate space is the GLU premise; SwiGLU gates the EXPANDED
    FFN intermediate, this gates the COMPRESSED one. act is swish/gelu/xatlu, not
    necessarily a sigmoid -- which matters, because the bounded case is the only
    one where the gate can merely attenuate.

    WHAT IT DOES AND DOES NOT DO. It does not lift the ceiling: K stays in
    range(up_k) and the bottleneck is still r, since c_tilde is a function of c.
    It makes the score NONLINEAR in x_j rather than bilinear, at a fixed r. Worth
    testing because it is nearly free (one extra d_model x r projection) and
    absorption survives: q^T K = (up_k^T q)^T c_tilde, so attention is still
    computed in the latent space against the cached vector.

    The (1 + act(.)) form with down_g zero-init is EXACT at init: act(0) = 0 for
    swish and gelu, so the factor is exactly 1.

(2) MULTI-TAP UP-PROJECTION -- raises the ceiling at zero cache cost.

        K_j = A_0 . c_j + A_1 . c_{j-1} + ... + A_k . c_{j-k}

    A "tap" is one coefficient of a causal filter, one per lag. Each A_i is its own
    r -> k_out matrix, so keys span the union of their ranges: up to (k+1)*r
    dimensions instead of r. The past latents are ALREADY cached, so this costs
    nothing extra to store, and it stays absorbable term by term:
        q^T K_j = sum_i (A_i^T q)^T c_{j-i}
    Decode cost on the score path grows by (k+1)x; cache cost by zero.

    NOTE A SHARED up-projection fed a convolved latent, up_k . sum_i w_i c_{j-i},
    does NOT lift anything -- the result is still inside range(up_k). The separate
    per-tap matrices are the whole point.

    A_1..A_k are zero-init, so the module is EXACT at init.

DECODE PATH. Both work unchanged for training and full-sequence eval. Multi-tap
incremental decode needs the last k cached latents, which the KV cache already
holds -- but the shift here is written for a full sequence, so a decode
implementation is still owed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(name):
    """Unbounded GLU activations, all with act(0) = 0 so the (1 + act) gate is
    exact at a zero init. See XATLUGate for the bounded, signed alternative that
    is the DEFAULT here, and why boundedness matters when the gate output is
    cached."""
    if name in ("swish", "silu", "swiglu"):
        return F.silu
    if name in ("gelu", "geglu"):
        return F.gelu
    if name in ("relu", "reglu"):
        return F.relu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"unknown gate activation {name!r}")


class XATLUGate(nn.Module):
    """xATLU -- Expanded ArcTan Linear Unit, as the gate on a CACHED latent.

        G(z) = (atan(z)/pi + 1/2) * (1 + 2*alpha) - alpha      gate in [-a, 1+a]
        factor = 2 * G(z)                                      factor(0) = 1 exactly

    G(0) = 1/2 for ANY alpha, since 0.5*(1+2a) - a = 0.5. So `2*G` is exactly 1 at
    a zero input regardless of the learned expansion, which is what makes the
    install exact at init without constraining alpha.

    WHY BOUNDED AND SIGNED IS THE RIGHT CHOICE *HERE*, unlike in an FFN GLU:
      * the gate output is CACHED. An unbounded gate (swish, gelu) lets cached
        values grow without limit, eating bf16 headroom and wrecking any later
        cache quantisation. A bounded gate keeps the cache well-scaled by
        construction. Nothing is stored in an FFN, so this argument does not
        appear there and unbounded gates are fine.
      * the expanded range is signed: with alpha > 0 the factor reaches below 0,
        so a token can FLIP a latent direction, not merely attenuate it. Read as
        modulating the latent subspace per token rather than masking it.

    alpha is learnable per channel and zero-init, so the gate starts in the plain
    [0,1] arctan range and expands only if training pays for it.
    """

    def __init__(self, r, device=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(r, device=device,
                                              dtype=torch.float32))

    def forward(self, z):
        a = self.alpha
        g = (torch.atan(z) / math.pi + 0.5) * (1.0 + 2.0 * a) - a
        return 2.0 * g


class GatedLatent(nn.Module):
    """Wraps a LatentKV's `down` so the cached latent is gated.

    Installed by replacing latent.down with this module, so nothing downstream
    changes: up_k/up_v still read one r-dimensional vector and absorption is
    untouched.
    """

    def __init__(self, down, act="xatlu"):
        super().__init__()
        self.down = down
        self.act_name = act
        d_in, r = down.in_features, down.out_features
        dev = down.weight.device
        # fp32 like every other new parameter here: the gate multiplies the
        # cached latent, so bf16 rounding on it is rounding on the cache.
        self.down_g = nn.Linear(d_in, r, bias=False, device=dev,
                                dtype=torch.float32)
        nn.init.zeros_(self.down_g.weight)
        if act in ("xatlu", "atan"):
            self.xatlu = XATLUGate(r, device=dev)   # factor(0) = 1 exactly
            self.act = None
        else:
            self.xatlu = None
            self.act = _act(act)                    # act(0)=0 -> 1+act = 1

    def forward(self, x):
        c = self.down(x)
        g = self.down_g(x.float())
        factor = self.xatlu(g) if self.xatlu is not None else 1.0 + self.act(g)
        return c * factor.to(c.dtype)


class MultiTapUp(nn.Module):
    """Wraps a LatentKV's `up_k` (or `up_v`) with k extra causal taps.

    out_j = A_0 c_j + sum_{i=1..k} A_i c_{j-i},  A_1..A_k zero-init.
    """

    def __init__(self, up, taps=1):
        super().__init__()
        assert taps >= 1, "taps counts the EXTRA lags; 0 means use the plain up"
        self.up = up
        self.taps = taps
        r, out = up.in_features, up.out_features
        dev = up.weight.device
        self.extra = nn.ModuleList()
        for _ in range(taps):
            lin = nn.Linear(r, out, bias=False, device=dev, dtype=torch.float32)
            nn.init.zeros_(lin.weight)         # exact at init
            self.extra.append(lin)

    def forward(self, c):
        out = self.up(c)
        cf = c.float()
        for i, lin in enumerate(self.extra, start=1):
            # causal shift by i along the sequence axis, zero-filled at the front
            shifted = F.pad(cf, (0, 0, i, 0))[..., : cf.shape[-2], :]
            out = out + lin(shifted).to(out.dtype)
        return out


def install_latent_ext(model, gate=None, taps=0, verbose=True):
    """Add the gate and/or taps to every LatentKV in the model.

    Returns (n_gated, n_tapped). Raises if there is no LatentKV, since silently
    doing nothing is how an arm ends up measuring the wrong thing.
    """
    from mercurius.surgery.transmla import LatentKV
    lats = [m for m in model.modules() if isinstance(m, LatentKV)]
    if not lats:
        raise SystemExit(
            "--mla-gate / --mla-taps found no LatentKV: MLA is not installed, so "
            "there is no latent to extend. Pass --mla-dc/--mla-groups.")
    n_g = n_t = 0
    for lat in lats:
        if gate:
            lat.down = GatedLatent(lat.down, act=gate); n_g += 1
        if taps:
            lat.up_k = MultiTapUp(lat.up_k, taps)
            lat.up_v = MultiTapUp(lat.up_v, taps)
            n_t += 1
    if verbose:
        bits = []
        if gate:
            bits.append(f"gated latent ({gate}, zero-init so exact at init) on "
                        f"{n_g} layers")
        if taps:
            bits.append(f"{taps} extra causal tap(s) on up_k/up_v of {n_t} layers "
                        f"(ceiling r -> up to {taps+1}r, cache unchanged)")
        print("  MLA latent ext: " + "; ".join(bits), flush=True)
    return n_g, n_t


# ---------------------------------------------------------------- (3) conv MLA
class DepthwiseCausalConv(nn.Module):
    """Depthwise causal conv over the sequence axis, IDENTITY-initialised.

    out_j[d] = sum_{a=0..k-1} w[d, a] * x[j - (k-1-a)][d]

    w is initialised to a delta at lag 0 (the last column), so the module is
    EXACTLY the identity at install and the surgery stays function-preserving --
    the same discipline as GatedLatent and MultiTapUp above. Note the contrast
    with those two: they are ZERO-init because they are additive/multiplicative
    corrections, while a conv must be delta-init because it REPLACES the signal.

    fp32 throughout. Per #31.4, MatryoshkaKV's full-rank row loses ~2 points on
    HellaSwag and PIQA purely to a bf16 round trip through its projections.

    CACHE COST IS O(1) IN SEQUENCE LENGTH. The conv needs the last k-1 inputs,
    which is a fixed-size rolling buffer per layer exactly like the conv state
    Qwen3.5's own GDN layers already carry (linear_conv_kernel_dim = 4). It is
    NOT a per-token cache entry, so it does not scale with context.
    """

    def __init__(self, d, k, device=None):
        super().__init__()
        assert k >= 2, "k=1 is pointwise, i.e. a no-op here"
        self.d, self.k = d, k
        w = torch.zeros(d, k, device=device, dtype=torch.float32)
        w[:, -1] = 1.0                       # delta at lag 0 => exact identity
        self.w = nn.Parameter(w)

    def forward(self, x):
        lead, T, d = x.shape[:-2], x.shape[-2], x.shape[-1]
        xt = x.reshape(-1, T, d).transpose(1, 2)          # (B, d, T)
        xt = F.pad(xt, (self.k - 1, 0))                   # causal: pad left only
        out = F.conv1d(xt, self.w.unsqueeze(1), groups=self.d)
        return out.transpose(1, 2).reshape(*lead, T, d)


class ConvDown(nn.Module):
    """Depthwise causal conv on the INPUT to a LatentKV's `down`.

        c_j = down( sum_a v_a (*) x_{j-a} ) = sum_a down diag(v_a) x_{j-a}

    This is the half that does something a pointwise down-projection cannot: the
    latent bottleneck now compresses a temporal WINDOW of the residual stream
    instead of a single position. Whether a window resolves better than a point
    at fixed d_c is the open question this arm exists to answer.
    """

    def __init__(self, down, k):
        super().__init__()
        self.down = down
        self.conv = DepthwiseCausalConv(down.in_features, k,
                                        device=down.weight.device)

    def forward(self, x):
        return self.down(self.conv(x.float()).to(x.dtype))


class ConvUp(nn.Module):
    """Depthwise causal conv on the CACHED LATENT, before `up_k` / `up_v`.

        K_j = up( sum_b w_b (*) c_{j-b} ) = sum_b up diag(w_b) c_{j-b}

    DOES NOT LIFT THE RANK CEILING, and that is the point of keeping it separate
    from MultiTapUp: every lag's effective matrix `up diag(w_b)` shares
    range(up), so keys stay in that same r-dimensional subspace. Multi-tap's
    independent A_i span up to (k+1)r. So this costs k*r parameters and buys
    temporal resolution; taps cost k*r*out and buy capacity. Orthogonal axes.

    Free at decode: the past latents are already in the cache, exactly as for
    MultiTapUp.
    """

    def __init__(self, up, k):
        super().__init__()
        self.up = up
        self.conv = DepthwiseCausalConv(up.in_features, k,
                                        device=up.weight.device)

    def forward(self, c):
        return self.up(self.conv(c.float()).to(c.dtype))


def install_mla_conv(model, k=4, where="both", verbose=True):
    """Depthwise causal convs on the MLA latent path. Exact at init.

    where = "pre"    conv before `down` only   (latent sees a window of x)
            "latent" conv before up_k/up_v only (temporal mix of cached latents)
            "both"   both, which is what was asked for

    NOTE these RENAME what they wrap (down -> down.down, up_k -> up_k.up), so a
    rebuild MUST install them before loading a checkpoint -- same constraint as
    GatedLatent and MultiTapUp, and the same reason #23 exists. They also collide
    with those two on the wrapped attribute name, so conv and gate/taps arms are
    mutually exclusive until someone writes a combined wrapper.
    """
    from mercurius.surgery.transmla import LatentKV
    lats = [m for m in model.modules() if isinstance(m, LatentKV)]
    if not lats:
        raise SystemExit(
            "--mla-conv found no LatentKV: MLA is not installed, so there is no "
            "latent path to convolve. Pass --mla-dc/--mla-groups.")
    if where not in ("pre", "latent", "both"):
        raise ValueError(f"unknown --mla-conv-where {where!r}")
    n_pre = n_lat = 0
    for lat in lats:
        if where in ("pre", "both"):
            lat.down = ConvDown(lat.down, k); n_pre += 1
        if where in ("latent", "both"):
            lat.up_k = ConvUp(lat.up_k, k)
            lat.up_v = ConvUp(lat.up_v, k)
            n_lat += 1
    if verbose:
        bits = []
        if n_pre:
            bits.append(f"pre-down conv k={k} on {n_pre} layers (latent compresses "
                        f"a window of x, not a point)")
        if n_lat:
            bits.append(f"pre-up conv k={k} on up_k/up_v of {n_lat} layers "
                        f"(temporal mix of cached latents; rank ceiling UNCHANGED)")
        print("  MLA conv: " + "; ".join(bits), flush=True)
    return n_pre, n_lat
