"""Per-channel residual-branch output scale, identity-initialized.

LayerScale (Touvron et al., CaiT, arXiv:2103.17239) is a learnable per-channel
diagonal on each residual BRANCH OUTPUT, before the add:

    x_{l+1} = x_l + diag(lambda) * branch(norm(x_l))

CaiT initializes it small (0.1 down to 1e-6 by depth) so a from-scratch deep
network starts near the identity function. That init is wrong here: our branches
are REPLACEMENTS (KDA for GDN, MLA for full KV), not additions. Zeroing a
replaced branch does not recover the teacher, it deletes the token mixer -- and
it throws away the surgical init (GDN tiling, whitened SVD) that we engineered.
MOHAWK makes the same call in the other direction, opening its Mamba-2 gate to 1
specifically to cancel the gate at init.

So: identity init. In the zero-centered `(1 + lam)` convention the model already
uses for its norms, that is lam = 0, which makes install bit-exact -- (1 + 0.0)
is exactly 1.0 in every dtype, so the multiply is a true no-op and the install
can be verified against the unmodified model.

What it buys at identity init is NOT stability -- it is capacity of a shape LoRA
cannot reach. For a branch ending in a Linear W,

    diag(1 + lam) @ W  -  W  =  diag(lam) @ W

is generally FULL RANK, so a rank-r adapter on that Linear cannot express it.
This is exactly the magnitude component of DoRA (Liu et al., arXiv:2402.09353).
It therefore only earns its keep where the branch's output Linear is LoRA-only:
if that weight is trained densely, dense already reaches any row scaling and
lam is redundant.

Placement note: install AFTER inject_lora, so lam scales (base + LoRA delta)
rather than the frozen base alone.

lam is kept in fp32 for the same reason the norm gains are: at lam ~ 0.5 the
bf16 ULP is 3.91e-3 while an Adam step at 3e-5 is ~3e-5, which would round to
exactly zero. 49,152 params in fp32 is 197 KB.

Folding: diag(1 + lam) @ W is a row scaling of the output Linear, so lam merges
into that weight before quantization. It widens the per-row dynamic range of
that weight, which is the same quantization caution Stage A raises for (1 + w).
"""
import torch
import torch.nn as nn

# the three residual-branch output projections, one per branch type
BRANCH_OUTPUTS = ("mlp.down_proj", "self_attn.o_proj", "linear_attn.out_proj")


class ScaledOutput(nn.Module):
    """Wraps a branch's output module with a per-channel (1 + lam) scale."""

    def __init__(self, inner: nn.Module, width: int):
        super().__init__()
        self.inner = inner
        dev = next(inner.parameters()).device
        self.ls_lambda = nn.Parameter(torch.zeros(width, device=dev, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        y = self.inner(*args, **kwargs)
        # (1 + 0.0) is exactly 1.0 in bf16, so this is bit-exact at init
        return y * (1.0 + self.ls_lambda).to(y.dtype)


def _out_width(mod):
    if isinstance(mod, nn.Linear):
        return mod.out_features
    base = getattr(mod, "base", None)          # LoRALinear
    if isinstance(base, nn.Linear):
        return base.out_features
    return None


def install_layerscale(model, targets=BRANCH_OUTPUTS, verbose=True):
    """Wrap each residual-branch output projection. Returns (count, params)."""
    hit, total = [], 0
    seen = set()
    for mod_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            # endswith, not substring: a LoRA-wrapped target exposes its frozen
            # base at "<target>.base", which CONTAINS the target string. A
            # substring test wraps both, applying two composed per-channel
            # scales to the same branch -- still identity at init, so it does
            # not break exactness, but it doubles the parameter count and the
            # inner copy scales only the base while the outer scales base+LoRA.
            # Measured: 96 wraps where 48 were intended.
            if not any(full.endswith(t) for t in targets):
                continue
            if id(child) in seen or isinstance(child, ScaledOutput):
                continue
            width = _out_width(child)
            if width is None:
                continue
            seen.add(id(child))
            setattr(parent, child_name, ScaledOutput(child, width))
            hit.append(full)
            total += width
    if verbose:
        by_kind = {}
        for f in hit:
            k = next(t for t in targets if t in f)
            by_kind[k] = by_kind.get(k, 0) + 1
        print(f"  LayerScale installed on {len(hit)} branch outputs "
              f"({', '.join(f'{c}x {k.split('.')[-1]}' for k, c in by_kind.items())}); "
              f"{total:,} params, identity init, fp32 -- bit-exact no-op at install")
    return len(hit), total
