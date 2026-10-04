"""Give every query head its own key, at no cache and no FLOP cost.

The MLA conversion reads its output widths off the original k_proj/v_proj, so it
kept GQA's 2 kv heads and only compressed: 8 query heads share 2 key subspaces.
TransMLA's actual claim is that GQA's replicated K/V is rank-deficient and a
latent lets each head have its own key at the same cache cost. Four heads forced
to share one key is interference by construction, and interference is the one
thing measurably broken here (multi-needle EM 98.8% -> 79-85%).

WHY THIS IS NOT IMPLEMENTED BY WIDENING up_k. Widening it to n_heads*head_dim
would be the literal reading, but the stock attention forward derives its view
shape from config.num_key_value_heads and then calls repeat_kv, so a widened k
would be replicated a second time into the wrong shape. The algebra offers a
cleaner route:

    score_h = q_h^T (W_h c) = ((W_h)^T q_h)^T c

so giving head h its own key matrix W_h is the same as giving it its own linear
map on q. A per-head block-diagonal transform on the query side is therefore
equivalent in expressiveness, changes no tensor shapes, needs no change to the
attention forward, and stays compatible with MLA absorption (it folds into the
query, which is exactly where absorption wants it).

Cost: n_heads * head_dim^2 = 8 * 256^2 = 524,288 per layer, 3.15 M over 6 layers.
Cache unchanged. FLOPs unchanged in attention -- the 8 replicated heads are
already materialised, this only stops forcing them to be equal.

Identity at init: R_h = I for every head, so the wrapper is a no-op until trained.

q_proj's output is laid out PER HEAD, [q_0 | g_0 | q_1 | g_1 | ...], each block
head_dim wide: the stock forward does
    q_proj(x).view(..., -1, head_dim * 2).chunk(2, dim=-1)
so the transform must touch the q block of each head ONLY.

FIXED 2026-09-21. This used to split the whole tensor in half with
out.chunk(2, dim=-1), which assumed [all q | all gates]. With that layout the
"query" half was q_0, g_0, q_1, g_1, ... for the first n_heads/2 heads: R_h for odd
h rotated OUTPUT GATES, and heads n_heads/2 .. n_heads-1 got no map at all.
Identity init hid it (a permuted identity is still the identity), so every
per-head-q arm before this date, gdn2phq included, trained a different and
half-broken mechanism.
"""
import torch
import torch.nn as nn


class PerHeadQ(nn.Module):
    def __init__(self, base: nn.Module, n_heads: int, head_dim: int):
        super().__init__()
        self.base = base
        self.n_heads, self.head_dim = n_heads, head_dim
        dev = next(base.parameters()).device
        self.R = nn.Parameter(
            torch.eye(head_dim, device=dev, dtype=torch.float32)
            .unsqueeze(0).repeat(n_heads, 1, 1))

    def forward(self, x):
        out = self.base(x)
        s = out.shape[:-1]
        qg = out.view(*s, self.n_heads, 2 * self.head_dim)
        q, gate = qg.split(self.head_dim, dim=-1)  # per head: query | gate
        q = torch.einsum("...hd,hde->...he", q.float(), self.R).to(out.dtype)
        return torch.cat([q, gate], dim=-1).reshape(*s, -1)


@torch.no_grad()
def install_per_head_q(model, verbose=True):
    """Wrap every self_attn q_proj. Applied LAST, after adapters and MLA."""
    cfg = model.config.text_config if hasattr(model.config, "text_config") \
        else model.config
    trunk = model.model.language_model if hasattr(model.model, "language_model") \
        else model.model
    n_heads = cfg.num_attention_heads
    head_dim = cfg.head_dim
    n = 0
    for layer in trunk.layers:
        sa = getattr(layer, "self_attn", None)
        if sa is None or not hasattr(sa, "q_proj") or isinstance(sa.q_proj, PerHeadQ):
            continue
        sa.q_proj = PerHeadQ(sa.q_proj, n_heads, head_dim)
        n += 1
    if verbose and n:
        print(f"  per-head query maps on {n} attention layers: "
              f"{n * n_heads * head_dim ** 2 / 1e6:.2f} M params, identity at init "
              f"(each query head gets its own effective key)", flush=True)
    return n


# ---------------------------------------------------------------- RoPE-commuting maps
# FIXED 2026-10-03 (#67). The equivalence this module rests on -- a map on q_h is the
# same as a per-head key, score_h = q_h^T W_h k = (W_h^T q_h)^T k -- holds only when no
# position-dependent transform sits between them. Under partial RoPE (dial c0) the
# first `rotary_dim` dims of q are rotated AFTER this map, so the score is
#     q^T R Rot(Delta) k,  not  q^T Rot(Delta) W k,
# and the two agree for all Delta only if R commutes with Rot(Delta). The unconstrained
# R trained in c0-long75 did not (relative commutator 0.42-0.53, 12.8% rotary<->NoPE
# mixing): it re-routed query content across RoPE frequencies, fitted only on offsets
# <= the training length, and long-range multi-key retrieval degraded (#66.2). This is
# exactly why MLA keeps RoPE decoupled from absorption (TransMLA, DeepSeek).
# The commutant of the RoPE rotations (rotate_half layout: pair (i, i + rotary_dim/2)
# turns by theta_i) is: on each pair a 2x2 [[a, -b], [b, a]] (a complex scalar), no
# mixing between pairs, none between rotary and NoPE dims; the NoPE block is free.
@torch.no_grad()
def project_rope_commute_(model, rotary_dim):
    """Project every PerHeadQ.R onto the RoPE commutant, in place. Idempotent; I -> I."""
    half = rotary_dim // 2
    i = torch.arange(half)
    j = i + half
    n = 0
    for m in model.modules():
        if not (isinstance(m, PerHeadQ) or getattr(m, "_phq_post", False)):
            continue
        R = m.R.data                                   # (H, D, D), q' = R^T q
        a = (R[:, i, i] + R[:, j, j]) / 2
        b = (R[:, i, j] - R[:, j, i]) / 2
        R[:, :rotary_dim, :] = 0
        R[:, :, :rotary_dim] = 0
        R[:, i, i] = a
        R[:, j, j] = a
        R[:, i, j] = b
        R[:, j, i] = -b
        n += 1
    return n


@torch.no_grad()
def tie_rope_norm_deltas_(model, rotary_dim, ref):
    """q_norm / k_norm gains sit before RoPE too: a per-dim gain commutes with the pair
    rotation only if both dims of a pair carry the same gain. The ORIGINAL gains are
    kept as they are (trained at full length); the TRAINED change is tied per pair.
    `ref` maps parameter name -> its value when tying started."""
    half = rotary_dim // 2
    for name, p in model.named_parameters():
        if name in ref:
            d = p.data[:rotary_dim] - ref[name][:rotary_dim]
            avg = (d[:half] + d[half:]) / 2
            p.data[:half] = ref[name][:half] + avg
            p.data[half:rotary_dim] = ref[name][half:rotary_dim] + avg


def rope_norm_refs(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and (n.endswith("q_norm.weight") or n.endswith("k_norm.weight"))}


# ---------------------------------------------------------------- post-norm placement
# FIXED 2026-10-03 (#67.1). Even in the commutant, a map placed in q_proj sits BEFORE
# q_norm, whose per-dim gains (1 + w) are strongly unequal within rotary pairs in the
# original model (ratio p10 0.73 / p90 1.31, worst 132x), so the transform reaching RoPE
# is D R^T, which commutes with the rotation only for a D-conjugated R. The map belongs
# immediately BEFORE RoPE, i.e. AFTER q_norm: then "commutes with RoPE" is exactly the
# condition for being a per-head key map. Registered on the q_norm module itself
# (param `q_norm.R`), so no existing parameter is renamed.
def _phq_post_forward(self, x):
    y = type(self).forward(self, x)                      # (..., H, D) normalized query
    return torch.einsum("...hd,hde->...he", y.float(), self.R).to(y.dtype)


@torch.no_grad()
def install_per_head_q_post(model, verbose=True):
    import types
    cfg = model.config.text_config if hasattr(model.config, "text_config") else model.config
    trunk = model.model.language_model if hasattr(model.model, "language_model") \
        else model.model
    H, D = cfg.num_attention_heads, cfg.head_dim
    n = 0
    for layer in trunk.layers:
        sa = getattr(layer, "self_attn", None)
        if sa is None or not hasattr(sa, "q_norm") or hasattr(sa.q_norm, "R"):
            continue
        qn = sa.q_norm
        dev = qn.weight.device
        qn.R = nn.Parameter(torch.eye(D, device=dev, dtype=torch.float32)
                            .unsqueeze(0).repeat(H, 1, 1))
        qn._phq_post = True
        qn.forward = types.MethodType(_phq_post_forward, qn)
        n += 1
    if verbose and n:
        print(f"  per-head query maps AFTER q_norm (right before RoPE) on {n} attention "
              f"layers: {n * H * D * D / 1e6:.2f} M params, identity at init", flush=True)
    return n
