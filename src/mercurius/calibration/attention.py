"""Attention-weighted input covariance for the MLA decomposition.

CARE whitens with C = X^T X, the covariance of the hidden states entering the
attention layer. That minimizes reconstruction error of K and V as matrices.
But the quantity that actually reaches the residual stream is

    O_h = P_h V = P_h X W_v^T          (P_h row-stochastic attention)

so the error that matters for V is

    ||dO_h||_F^2 = || P_h X dW_v^T ||_F^2 = tr( dW_v (P_h X)^T (P_h X) dW_v^T )

which is the SAME quadratic form with a DIFFERENT covariance: the covariance of
attention-weighted activations, not raw ones. Nothing about the whitening,
Cholesky or SVD changes -- only the matrix fed to them. That is the whole idea.

Under GQA each KV head g is read by several query heads, and the errors add, so

    C_v[g] = sum_{h in g} (P_h X)^T (P_h X)

and since TransMLA shares ONE down-projection across K and V and across KV
heads, what the whitening can actually consume is a single d x d matrix per
layer: the sum over all heads. A principled shared metric is C_x + C_v -- the
K-path error (for which X^T X is the standing proxy) plus the V-path error.

The K side is NOT exact here. Errors in K pass through the softmax, so the
first-order term is dP ~ P*(dlogits - E_P[dlogits]), not a quadratic form in
dW_k. X^T X remains a proxy for that path; only the V path is derived.

Both covariances are accumulated in the SAME pass over the SAME samples, so an
A/B between them differs only in the statistic, never in the calibration draw.

Cost: eager attention at seq 512 materializes (8, 512, 512) probabilities per
layer -- 4 MB in bf16, transient. The fp64 accumulation is ~8x the plain one
(one term per query head), which measures in the tens of seconds, not hours.
"""
import sys, torch
from mercurius.calibration.care import get_trunk


def _force_eager(model):
    """Attention probabilities are only observable on the eager path."""
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "eager"
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass
    return model


@torch.no_grad()
def collect_attn_covariances(model, ids, n_samples=256, seq=512, seed=0,
                             verbose=True):
    """Return {layer: {"x": C_x, "v": C_v}} -- raw and attention-weighted.

    C_x reproduces run_care.collect_covariances exactly (same hook point, same
    fp64 accumulation, same normalization) so the pair is directly comparable.
    """
    _force_eager(model)
    trunk = get_trunk(model)
    attn_idx = [i for i, l in enumerate(trunk.layers) if hasattr(l, "self_attn")]
    d = trunk.layers[0].input_layernorm.weight.shape[0]

    cx = {i: torch.zeros(d, d, dtype=torch.float64, device="cuda") for i in attn_idx}
    cv = {i: torch.zeros(d, d, dtype=torch.float64, device="cuda") for i in attn_idx}
    counts = {i: 0 for i in attn_idx}
    xs = {}

    hooks = []
    def mk(i):
        def hook(mod, inp, out):
            xs[i] = inp[0].detach()          # (B, T, d) entering k_proj
        return hook
    for i in attn_idx:
        hooks.append(trunk.layers[i].self_attn.k_proj.register_forward_hook(mk(i)))

    g = torch.Generator().manual_seed(seed)
    n_layers_seen = 0
    for s in range(n_samples):
        off = int(torch.randint(0, len(ids) - seq - 1, (1,), generator=g))
        out = model(input_ids=ids[off:off + seq].unsqueeze(0).cuda(),
                    output_attentions=True, logits_to_keep=1)
        attns = out.attentions
        if attns is None or all(a is None for a in attns):
            raise RuntimeError(
                "output_attentions returned nothing -- the eager path is not "
                "active, so attention-weighted covariance cannot be collected. "
                "Check transformers' attn_implementation dispatch for this model.")

        # attns carries one entry per ATTENTION-PRODUCING layer, not per model
        # layer. This model is hybrid -- 24 layers, 6 with self_attn -- so the
        # tuple has 6 entries and attn_idx is [3,7,11,15,19,23]. Indexing it by
        # the model layer index silently returns the WRONG layer's attention for
        # i <= 5 and IndexErrors above that. Some versions instead return a
        # full-length tuple with None in the linear-attention slots, so handle
        # both rather than assuming.
        if len(attns) == len(trunk.layers):
            pos = {i: i for i in attn_idx}
        elif len(attns) == len(attn_idx):
            pos = {i: k for k, i in enumerate(attn_idx)}
        else:
            raise RuntimeError(
                f"cannot map layers to attentions: {len(attns)} attention "
                f"entries for {len(trunk.layers)} layers and {len(attn_idx)} "
                f"attention layers {attn_idx}")

        for i in attn_idx:
            x = xs[i]                                   # (B, T, d)
            P = attns[pos[i]]                           # (B, H, T, T)
            if P is None:
                raise RuntimeError(f"layer {i} returned no attention weights")
            n_layers_seen = max(n_layers_seen, 1)
            xf = x.reshape(-1, x.shape[-1])
            cx[i] += (xf.double().T @ xf.double())
            # Centre P by its column (query) mean before forming P_h X.
            # WITHOUT THIS THE STATISTIC IS USELESS: attention sinks put most
            # mass on token 0, so P is near rank-1 and C_v collapses with it --
            # measured effective rank 1.01 of 64 at a 60% sink, 1.00 at 90%,
            # with the top eigenvalue holding 99.7% of the energy. The entire
            # latent budget would go to reconstructing the sink token's value
            # vector, and the failure would look like a bug in this file rather
            # than a bug in the objective.
            # Centring fixes it (1.01 -> 4.23, and invariant to sink fraction;
            # a local-window pattern is unchanged, 27.66 -> 27.62). Justified:
            # a component of dV common to every query position is a bias-like
            # offset in the residual stream, the least harmful error mode.
            Pc = P - P.mean(dim=-2, keepdim=True)
            # (B,H,T,T) @ (B,1,T,d) -> (B,H,T,d): P_h X for every query head
            px = torch.matmul(Pc.to(x.dtype), x.unsqueeze(1))
            px = px.reshape(-1, px.shape[-1]).double()  # (B*H*T, d), heads summed
            cv[i] += px.T @ px
            counts[i] += xf.shape[0]
        xs.clear()
        if verbose and (s + 1) % 64 == 0:
            print(f"    calibrated {s+1}/{n_samples}", flush=True)
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()
    for i in attn_idx:
        n = max(counts[i], 1)
        cx[i] /= n
        cv[i] /= n            # same denominator: cv carries the H-head sum
    if verbose:
        print(f"    covariance from {counts[attn_idx[0]]:,} token vectors "
              f"({len(attn_idx)} attention layers)", flush=True)
    return {i: {"x": cx[i].float().cpu(), "v": cv[i].float().cpu()}
            for i in attn_idx}


def mix(covs, mode):
    """Build the covariance dict convert_to_mla consumes.

    mode:
      'x'     -- CARE baseline, X^T X
      'v'     -- attention-weighted only
      'sum'   -- raw C_x + C_v
      'norm'  -- trace-normalized half-and-half (scale-free version of 'sum')
    """
    out = {}
    for i, c in covs.items():
        cx, cv = c["x"].double(), c["v"].double()
        if mode == "x":
            m = cx
        elif mode == "v":
            m = cv
        elif mode == "sum":
            m = cx + cv
        elif mode == "norm":
            tx = cx.diagonal().sum().clamp_min(1e-30)
            tv = cv.diagonal().sum().clamp_min(1e-30)
            m = 0.5 * (cx / tx + cv / tv) * tx      # keep C_x's scale
        else:
            raise ValueError(mode)
        m = 0.5 * (m + m.T)                          # kill accumulated asymmetry
        out[i] = m.float()
    return out
