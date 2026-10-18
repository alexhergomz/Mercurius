"""VeRA adapters (Kopiczko et al., ICLR 2024): dW = diag(b) B diag(d) A with A, B random, frozen and SHARED by every
adapted layer, so a layer trains rank + out_features values. Rank is nearly free; Mercurius uses 1024.

The shared pair lives once in a non-persistent bank (regenerated from its seed), so checkpoints carry only the
trained d and b vectors, and merging writes the dense delta into the base weight for export.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class VeRABank(nn.Module):
    def __init__(self, rank, max_in, max_out, seed=0, device=None, dtype=torch.float32):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        A = torch.empty(rank, max_in)
        B = torch.empty(max_out, rank)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
        nn.init.kaiming_uniform_(B, a=math.sqrt(5), generator=g)
        self.register_buffer("A", A.to(device, dtype), persistent=False)
        self.register_buffer("B", B.to(device, dtype), persistent=False)
        self.seed, self.rank = seed, rank


class VeRALinear(nn.Module):
    def __init__(self, base: nn.Linear, bank: VeRABank, d_init=0.1):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self._bank = [bank]                                     # a list: not registered as a child module
        dev = base.weight.device
        # fp32 masters: a bf16 vector would round small Adam steps away; the products run in the bank's dtype
        self.vera_d = nn.Parameter(torch.full((bank.rank,), float(d_init), device=dev, dtype=torch.float32))
        self.vera_b = nn.Parameter(torch.zeros(base.out_features, device=dev, dtype=torch.float32))

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    def factors(self):
        bank = self._bank[0]
        dt = bank.A.dtype
        A = bank.A[:, :self.in_features] * self.vera_d.to(dt).unsqueeze(1)                    # (rank, in)
        B = bank.B[:self.out_features] * self.vera_b.to(dt).unsqueeze(1)                      # (out, rank)
        return A, B

    def adapter(self, x):
        A, B = self.factors()
        return (x.to(A.dtype) @ A.T) @ B.T

    def delta_weight(self):
        A, B = self.factors()
        return B.float() @ A.float()

    def forward(self, x):
        return self.base(x) + self.adapter(x).to(x.dtype)


def inject_vera(model, targets, rank=1024, d_init=0.1, seed=0, log=print):
    """Wrap every nn.Linear whose qualified name ends with one of `targets`. Returns the wrapped names."""
    found = []
    for name, parent in list(model.named_modules()):
        for cname, child in list(parent.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                found.append((parent, cname, child, full))
    if not found:
        return []
    ref = found[0][2].weight
    bank = VeRABank(rank, max(c.in_features for *_, c, _ in found), max(c.out_features for *_, c, _ in found),
                    seed, ref.device, torch.float32 if ref.dtype == torch.float32 else torch.bfloat16)
    model.vera_bank = bank                                      # registered once, moves with the model
    for parent, cname, child, _ in found:
        setattr(parent, cname, VeRALinear(child, bank, d_init))
    log(f"  VeRA rank {rank} on {len(found)} projections: "
        f"{sum(rank + c.out_features for *_, c, _ in found) / 1e6:.2f} M trainable")
    return [f for *_, f in found]


@torch.no_grad()
def merge_vera(model):
    """Fold every adapter into its base and unwrap it (for export). Returns the count."""
    n = 0
    for name, parent in list(model.named_modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, VeRALinear):
                w = child.base.weight
                w.copy_((w.float() + child.delta_weight()).to(w.dtype))
                setattr(parent, cname, child.base)
                n += 1
    if hasattr(model, "vera_bank"):
        del model.vera_bank
    return n
