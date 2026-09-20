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

q_proj emits [query | gate] concatenated (Qwen3.5 gates attention output with the
second half), so the transform must touch the first half ONLY. Applying it to the
whole tensor would silently rotate the gate as well.
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
        q, gate = out.chunk(2, dim=-1)              # query half | gate half
        s = q.shape[:-1]
        q = q.view(*s, self.n_heads, self.head_dim).float()
        q = torch.einsum("...hd,hde->...he", q, self.R)
        return torch.cat([q.reshape(*s, -1).to(out.dtype), gate], dim=-1)


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
