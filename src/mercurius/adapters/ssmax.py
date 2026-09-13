"""Scalable-Softmax (SSMax): per-head learnable attention logit scaling.

    SSMax(z_i) = n^{s z_i} / sum_j n^{s z_j} = softmax( (s * log n) * z )

`n` is the sequence length and `s` is learnable per head. The diagnosis it
addresses is "attention fading": as n grows, softmax over more keys drives its
maximum toward 0, so heads that should be sharply peaked on one retrieved token
spread out instead. Retrieval heads are exactly the sharply-peaked ones, so this
is the mechanism that should matter for our failure -- and it is the opposite
sign to blurring the teacher's attention, which is a published negative.

WHERE IT IS APPLIED, AND WHY HERE. Qwen3_5Attention builds the query as

    query_states, gate = chunk(q_proj(h).view(*input_shape, -1, head_dim*2), 2, -1)
    query_states = self.q_norm(query_states.view(B, T, H, D)).transpose(1, 2)

so q_proj's output is HALF query and HALF output gate (attn_output_gate=True) --
scaling q_proj would corrupt the gate. q_norm sees only the query, already
shaped (B, T, H, D), which gives us both the head axis (-2) and the sequence
length (shape[1]) without hooking the attention forward at all. Scaling the
query is mathematically identical to scaling the logits, since the attention
interface applies `scaling` to q @ k^T afterwards.

INITIALIZATION. s = 1 / log(n_ref) makes the factor s*log(n_ref) equal 1.0, so at
the reference length this is a no-op and the model starts where it was. Measured:
the residual is 4.77e-07 in float32 -- (1/log n)*log n is not exactly 1.0 in
floating point -- which is below bf16 epsilon (~8e-03), so in the bf16 model the
factor rounds to exactly 1.0 and it IS bit-exact there. Do not describe it as
bit-exact in fp32. Longer sequences sharpen, shorter ones soften, and s then learns.
That follows the same discipline as the rest of this project: an intervention
that begins as an identity and has to earn its departure from it.

48 parameters total (6 full-attention layers x 8 heads).
"""
import math
import torch
import torch.nn as nn


def install_ssmax(model, n_ref: int = 8192, verbose: bool = True):
    """Wrap q_norm on every full-attention layer. Returns a restore fn."""
    from mercurius.surgery.norm_fusion import get_trunk
    layers = get_trunk(model).layers
    undo = []
    n_par = 0
    for l in layers:
        sa = getattr(l, "self_attn", None)
        if sa is None or not hasattr(sa, "q_norm"):
            continue
        H = model.config.num_attention_heads
        dev = sa.q_norm.weight.device
        sa.ssmax_s = nn.Parameter(
            torch.full((H,), 1.0 / math.log(n_ref), dtype=torch.float32, device=dev))
        n_par += H
        orig = sa.q_norm.forward

        def wrapped(x, _o=orig, _sa=sa):
            out = _o(x)                       # (B, T, H, D)
            n = x.shape[1]
            if n <= 1:
                return out                    # log(1) = 0 would zero the query
            s = _sa.ssmax_s.to(out.dtype).view(1, 1, -1, 1)
            return out * (s * math.log(n))

        sa.q_norm.forward = wrapped
        undo.append((sa, orig))

    if verbose:
        print(f"  SSMax installed on {len(undo)} attention layers "
              f"({n_par} params, exact no-op at n={n_ref})", flush=True)

    def restore():
        for sa, orig in undo:
            sa.q_norm.forward = orig
            if hasattr(sa, "ssmax_s"):
                del sa.ssmax_s
    return restore


@torch.no_grad()
def attention_entropy(model, ids, n, layer_idx=None):
    """Mean attention entropy per layer at sequence length n.

    This is the quantity SSMax is supposed to control, so it is the direct test:
    if entropy climbs with n on the uninstalled model and flattens with SSMax,
    the mechanism is doing what it claims -- measurable without any training.

    Uses eager attention to get the weights back. O(n^2) per layer, so keep n
    small: at 4096 one layer's map is 0.25 GiB in bf16 and 0.5 GiB at the fp32
    softmax. Do not call this at 32768.
    """
    import torch.nn.functional as F
    from mercurius.surgery.norm_fusion import get_trunk
    trunk = get_trunk(model)
    x = ids[:n].unsqueeze(0).cuda()
    ents, hooks = {}, []

    def mk(i):
        def hook(mod, args, kwargs, out):
            q = kwargs.get("query_states", args[1] if len(args) > 1 else None)
            return out
        return hook

    # simplest reliable route: recompute logits from the module's own q/k
    for i, l in enumerate(trunk.layers):
        sa = getattr(l, "self_attn", None)
        if sa is None or (layer_idx is not None and i != layer_idx):
            continue
        cap = {}

        def pre(mod, args, kwargs, _c=cap):
            _c["h"] = args[0] if args else kwargs.get("hidden_states")
            return None
        hooks.append(sa.register_forward_pre_hook(pre, with_kwargs=True))
        ents[i] = cap

    model(input_ids=x, logits_to_keep=1)
    for h in hooks:
        h.remove()

    out = {}
    for i, cap in ents.items():
        sa = trunk.layers[i].self_attn
        h = cap["h"]
        B, T, _ = h.shape
        hs = (B, T, -1, sa.head_dim)
        q, _g = torch.chunk(sa.q_proj(h).view(B, T, -1, sa.head_dim * 2), 2, dim=-1)
        q = sa.q_norm(q.view(hs)).transpose(1, 2)
        k = sa.k_norm(sa.k_proj(h).view(hs)).transpose(1, 2)
        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        logits = (q.float() @ k.float().transpose(-1, -2)) * sa.scaling
        mask = torch.full((T, T), float("-inf"), device=logits.device).triu(1)
        p = F.softmax(logits + mask, dim=-1)
        out[i] = float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
        del q, k, logits, p
        torch.cuda.empty_cache()
    del x
    return out
