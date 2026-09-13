"""Stage B — transKDA: lift GDN's per-head scalar decay to KDA's per-channel diagonal.

Qwen3.5's GDN computes, per (batch, time, head):

    g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)        # (B, T, H)

KDA wants the same quantity per channel: (B, T, H, D). Broadcasting the scalar
across D reproduces GDN exactly -- verified at kernel level to 3.4e-6 relative.

Split into two independently checkable pieces so a failure localizes:

  B1  install_kda_kernel(): route the layer through chunk_kda with g broadcast
      at the call site. Mathematically identical; tiny patch surface.

  B2  lift_gate_params():   materialize the lifted parameters -- row-tile
      in_proj_a, broadcast A_log and dt_bias -- and assert they reproduce the
      scalar path's g bit-for-bit. These are what training later adapts.

For training, the plan expresses B2's channel structure as a zero-init LoRA on
top of the frozen tiled projection, so the lift stays exact at init while
costing ~4.7M params instead of ~200M at 9B scale.
"""
import torch
import torch.nn.functional as F
from fla.ops import chunk_kda, fused_recurrent_kda

import transformers.models.qwen3_5.modeling_qwen3_5 as qm


# ---------------------------------------------------------------- B1
def _expand_g(g, head_dim):
    """(B, T, H) scalar decay -> (B, T, H, D) diagonal decay."""
    if g.dim() == 4:
        return g
    return g.unsqueeze(-1).expand(*g.shape, head_dim).contiguous()


def install_kda_kernel(head_dim=128):
    """Route GDN layers through the KDA kernel with broadcast decay."""
    orig_chunk = qm.torch_chunk_gated_delta_rule
    orig_recur = qm.torch_recurrent_gated_delta_rule

    def chunk_shim(query, key, value, g, beta, **kw):
        kw.pop("cp_context", None)
        return chunk_kda(q=query, k=key, v=value,
                         g=_expand_g(g.float(), head_dim), beta=beta, **kw)

    def recur_shim(query, key, value, g, beta, **kw):
        kw.pop("cp_context", None)
        return fused_recurrent_kda(q=query, k=key, v=value,
                                   g=_expand_g(g.float(), head_dim), beta=beta, **kw)

    qm.torch_chunk_gated_delta_rule = chunk_shim
    qm.torch_recurrent_gated_delta_rule = recur_shim
    return orig_chunk, orig_recur


def restore_gdn_kernel(orig):
    qm.torch_chunk_gated_delta_rule, qm.torch_recurrent_gated_delta_rule = orig


# ---------------------------------------------------------------- B2
@torch.no_grad()
def lift_gate_params(layer, verify_with=None, atol=0.0):
    """Row-tile the decay projection so it emits D values per head instead of 1.

    in_proj_a.weight : (H, d_model) -> (H*D, d_model), each row repeated D times
    A_log, dt_bias   : (H,)         -> (H, D),         each scalar broadcast

    Returns (n_params_before, n_params_after, max_g_deviation).
    """
    la = layer.linear_attn
    H = la.A_log.shape[0]
    D = la.head_k_dim

    W = la.in_proj_a.weight.data                       # (H, d_model)
    W_tiled = W.repeat_interleave(D, dim=0).contiguous()   # (H*D, d_model)
    A_log_d = la.A_log.data.unsqueeze(-1).repeat(1, D).contiguous()   # (H, D)
    dt_bias_d = la.dt_bias.data.unsqueeze(-1).repeat(1, D).contiguous()

    dev = 0.0
    if verify_with is not None:
        x = verify_with
        a_scalar = F.linear(x, W)                                   # (..., H)
        g_scalar = -la.A_log.float().exp() * F.softplus(a_scalar.float() + la.dt_bias)

        a_diag = F.linear(x, W_tiled).unflatten(-1, (H, D))          # (..., H, D)
        g_diag = -A_log_d.float().exp() * F.softplus(a_diag.float() + dt_bias_d)

        dev = (g_diag - g_scalar.unsqueeze(-1)).abs().max().item()

    before = W.numel() + la.A_log.numel() + la.dt_bias.numel()
    after = W_tiled.numel() + A_log_d.numel() + dt_bias_d.numel()
    return before, after, dev, (W_tiled, A_log_d, dt_bias_d)


@torch.no_grad()
def verify_all_layers(model, seed=0):
    """Run B2's parameter lift on every GDN layer and report the worst deviation."""
    from mercurius.surgery.norm_fusion import get_trunk
    torch.manual_seed(seed)
    trunk = get_trunk(model)
    worst, tot_before, tot_after, n = 0.0, 0, 0, 0
    for layer in trunk.layers:
        if not hasattr(layer, "linear_attn"):
            continue
        d_model = layer.linear_attn.in_proj_a.weight.shape[1]
        x = torch.randn(4, 32, d_model, device=layer.linear_attn.in_proj_a.weight.device,
                        dtype=layer.linear_attn.in_proj_a.weight.dtype)
        b, a, dev, _ = lift_gate_params(layer, verify_with=x)
        worst = max(worst, dev)
        tot_before += b; tot_after += a; n += 1
    return n, tot_before, tot_after, worst
