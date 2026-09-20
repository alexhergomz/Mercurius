"""NF4 quantization of the converted model.

Applied after surgery, per the plan's ordering: all structural work happens in
high precision, then the base is quantized once and frozen, and only bf16
adapters train on top.

Skipped deliberately:
  * lm_head            -- tied to embed_tokens; quantizing it would quantize the
                          input embedding path too
  * embed_tokens       -- 254M of the 0.8B model; lookup, not matmul
  * in_proj_a          -- the decay projection Stage B lifted. Its LoRA delta is
                          the new capacity; quantizing the frozen base under it
                          is fine, but the gate is numerically sensitive
                          (cumulative log-decay), so it stays bf16 by default.
"""
import torch
import torch.nn as nn
import bitsandbytes as bnb

SKIP_DEFAULT = ("lm_head", "embed_tokens", "in_proj_a")


def _swap(parent, name, lin, compute_dtype):
    q = bnb.nn.Linear4bit(
        lin.in_features, lin.out_features, bias=lin.bias is not None,
        compute_dtype=compute_dtype, quant_type="nf4",
        compress_statistics=True,
    )
    q.weight = bnb.nn.Params4bit(lin.weight.data.to(torch.bfloat16),
                                 requires_grad=False, quant_type="nf4")
    if lin.bias is not None:
        q.bias = nn.Parameter(lin.bias.data.to(compute_dtype), requires_grad=False)
    setattr(parent, name, q)


def quantize_nf4(model, skip=SKIP_DEFAULT, compute_dtype=torch.bfloat16,
                 verbose=True):
    """Replace nn.Linear with NF4 Linear4bit in place. Returns (n_swapped, n_skipped)."""
    targets = []
    for mod_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if any(s in full for s in skip):
                continue
            targets.append((parent, child_name, child, full))

    for parent, child_name, child, _ in targets:
        _swap(parent, child_name, child, compute_dtype)

    model.cuda()
    n_lin = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    if verbose:
        print(f"  quantized {len(targets)} Linear -> NF4; {n_lin} left in full precision")
    return len(targets), n_lin


def param_bytes(model):
    tot = 0
    for p in model.parameters():
        if hasattr(p, "quant_state") and p.quant_state is not None:
            tot += p.numel() // 2 + 0        # 4 bits/elem, packed
        else:
            tot += p.numel() * p.element_size()
    return tot
