"""Stage A': the folded RMSNorms become ScaleNorm -- one scalar each.

Stage A folds every foldable RMSNorm gain into the projections that consume
it, leaving the norm with weight exactly 0, i.e. a gain of (1 + 0) = 1 in
Qwen's zero-centred parameterisation. A parameter-free RMSNorm IS ScaleNorm
with a fixed gain, so replacing it with a learnable SCALAR initialised at 1.0
is exactly function-preserving and completes the swap the README describes.

Why bother. --train-norms trained 174k per-channel gains; 164k of them live in
these 64 modules. As a scalar per norm that becomes 64 parameters: the layer
can still adapt its activation SCALE -- which is what shifts after linearising
attention and compressing the KV cache -- without a per-channel surface that a
150-step run can overfit. Per-channel gains are also exactly what stage A just
removed; re-learning them would undo the fold.

NOT converted, because for these the swap would NOT be function-preserving:
  * model.norm          -- its gain is live (lm_head is tied to embed_tokens,
                           so stage A could not fold it)
  * q_norm / k_norm     -- per-head-dim gains applied AFTER normalisation; they
                           do not commute with the per-head query map, and the
                           product g_q * g_k cannot be folded into R exactly
                           because R sits inside q_norm's own rms
  * linear_attn.norm    -- fused gated RMSNorm inside the KDA/GDN-2 layer
Together ~10k parameters, left per-channel and still trainable.
"""
import torch
import torch.nn as nn

from mercurius.models import fused as _fused


class ScaleNorm(nn.Module):
    """y = g * x / rms(x), with g a single learned scalar (fp32).

    fp32 like every other gain here: at g ~ 1 the bf16 ULP is 7.8e-3 and an
    Adam step at 2e-4 would round to zero (findings 4.3).
    """

    def __init__(self, eps=1e-6, init=1.0, device=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.tensor(float(init), dtype=torch.float32,
                                                device=device))

    def forward(self, x):
        # Routed through the SAME fused function as the zero-centred RMSNorm it
        # replaces (g enters as 1 + (g-1)), so the compiled kernel and its
        # reduction order are identical and the swap is bitwise exact at
        # g = 1. Calling a separate scalar kernel instead was not: different
        # reduction order, and over 32 layers that reached 0.32 in the logits.
        return _fused.rms(x, self.weight - 1.0, self.eps)

    def extra_repr(self):
        return f"scalar, eps={self.eps}"


@torch.no_grad()
def convert_to_scalenorm(model, atol=0.0, verbose=True):
    """Replace every FOLDED RMSNorm (weight == 0) in the decoder layers.

    A norm whose weight is not zero is left alone and counted: converting it
    would change the function, and silently averaging a per-channel gain into
    a scalar is exactly the kind of quiet damage this project refuses.
    """
    from mercurius.surgery.norm_fusion import get_trunk
    n, skipped = 0, []
    for i, layer in enumerate(get_trunk(model).layers):
        for name in ("input_layernorm", "post_attention_layernorm"):
            mod = getattr(layer, name, None)
            if mod is None or isinstance(mod, ScaleNorm):
                continue
            w = getattr(mod, "weight", None)
            if w is None:
                continue
            dev = w.device
            mx = w.detach().abs().max().item()
            if mx > atol:
                skipped.append((f"layers.{i}.{name}", mx))
                continue
            setattr(layer, name, ScaleNorm(eps=getattr(mod, "eps", 1e-6),
                                           init=1.0, device=dev))
            n += 1
    if verbose:
        print(f"  ScaleNorm: {n} folded RMSNorms -> one scalar each "
              f"({n} parameters, was {n * get_trunk(model).config.hidden_size:,})",
              flush=True)
        if skipped:
            print(f"  ScaleNorm: left {len(skipped)} unfolded norms per-channel "
                  f"(largest |gain-1| {max(s for _, s in skipped):.3g})", flush=True)
    return n, skipped
