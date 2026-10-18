"""Stage A — fold RMSNorm gains into consumer weights (RMSNorm -> ScaleNorm).

RMSNorm computes  y = (x / rms(x)) * gamma,  and the next Linear applies W:

    W @ (gamma * x_hat)  ==  (W @ diag(gamma)) @ x_hat

so gamma folds into the weight and the norm becomes parameter-free. A
parameter-free RMSNorm is identically sqrt(d) * x/||x||, i.e. ScaleNorm with a
fixed gain; making that gain learnable completes the swap.

Exactly function-preserving in exact arithmetic. Run in fp32 so the fold itself
does not introduce rounding -- this is why the plan does all surgery before
quantization.

Deliberately NOT fused (see report at bottom):
  * final model.norm  -- lm_head is tied to embed_tokens; folding would
    corrupt the input embedding path.
  * q_norm / k_norm   -- gamma is applied AFTER normalization, so it cannot
    fold backward into q_proj; and a per-dim scale does not commute with
    RoPE's rotation pairs, so it cannot fold forward either.
  * linear_attn.norm  -- foldable in principle, but irrelevant to the
    transforms this project performs.
"""
import torch


def get_trunk(model):
    lm = model
    for attr in ("model", "language_model"):
        if hasattr(lm, attr):
            lm = getattr(lm, attr)
    return lm


def fuse_norm_into(norm, consumers):
    """W' = W @ diag(g_eff); then reset the norm to identity.

    CRITICAL: Qwen3_5RMSNorm is ZERO-CENTERED --

        output = x/rms(x) * (1.0 + weight)      # weight initialized to zeros

    so the effective gain is (1 + weight), not weight, and the identity value
    for the parameter is 0.0, not 1.0. Folding `weight` directly and resetting
    to 1.0 destroys the model (measured: max logit delta 25.1, top-1 agreement
    3.9%). The class is registered as "RMSNormZeroCentered" upstream.
    """
    g_eff = 1.0 + norm.weight.data.detach().clone()
    for lin in consumers:
        assert lin.weight.shape[1] == g_eff.shape[0], (
            f"shape mismatch: {tuple(lin.weight.shape)} vs {tuple(g_eff.shape)}")
        lin.weight.data.mul_(g_eff.unsqueeze(0))
    norm.weight.data.zero_()          # (1 + 0) == 1 -> parameter-free
    return g_eff


def fuse_model(model, verbose=True):
    trunk = get_trunk(model)
    fused, skipped = [], []

    for i, layer in enumerate(trunk.layers):
        # --- input_layernorm -> token-mixer input projections ---
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            consumers = [la.in_proj_qkv, la.in_proj_z, la.in_proj_b, la.in_proj_a]
            kind = "GDN"
        else:
            sa = layer.self_attn
            consumers = [sa.q_proj, sa.k_proj, sa.v_proj]
            kind = "ATTN"
        g = fuse_norm_into(layer.input_layernorm, consumers)
        fused.append((f"layers.{i}.input_layernorm", kind, len(consumers),
                      g.abs().max().item(), g.abs().min().item()))

        # --- post_attention_layernorm -> MLP input projections ---
        g = fuse_norm_into(layer.post_attention_layernorm,
                           [layer.mlp.gate_proj, layer.mlp.up_proj])
        fused.append((f"layers.{i}.post_attention_layernorm", "MLP", 2,
                      g.abs().max().item(), g.abs().min().item()))

    skipped.append(("model.norm", "tied lm_head/embed_tokens"))
    for i, layer in enumerate(trunk.layers):
        if hasattr(layer, "self_attn"):
            skipped.append((f"layers.{i}.self_attn.q_norm", "post-norm gamma; RoPE non-commuting"))
            skipped.append((f"layers.{i}.self_attn.k_norm", "post-norm gamma; RoPE non-commuting"))
        else:
            skipped.append((f"layers.{i}.linear_attn.norm", "not needed by our transforms"))

    if verbose:
        print(f"fused {len(fused)} norms across {len(trunk.layers)} layers")
        print(f"skipped {len(skipped)} norms (by design)")
        print("\n  effective gain (1+w) dynamic range -- the quantization risk:")
        worst = sorted(fused, key=lambda r: -(r[3] / max(r[4], 1e-9)))[:5]
        for name, kind, n, gmax, gmin in worst:
            print(f"    {name:<44} {kind:<5} max {gmax:8.4f}  min {gmin:8.5f}"
                  f"  ratio {gmax/max(gmin,1e-9):9.1f}")
    return fused, skipped
