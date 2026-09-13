"""Minimal LoRA injection for nn.Linear.

Hand-rolled rather than peft: the KDA layer is a custom class carrying its own
gate LoRA, and wrapping it in peft complicates the save/load path we already
verified bit-exact. This is ~40 lines and keeps that path intact.

Zero-init B, so the adapted model starts exactly at the base model.
"""
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = (alpha or rank) / rank
        dev = base.weight.device
        dt = torch.bfloat16 if base.weight.dtype != torch.float32 else torch.float32
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        d = (x.to(self.lora_A.dtype) @ self.lora_A.T) @ self.lora_B.T
        return out + self.scale * d.to(out.dtype)


def inject_lora(model, rules, verbose=True):
    """rules: list of (substring, rank). First match wins; rank 0 = skip."""
    injected, total_new = [], 0
    seen = set()
    for mod_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if id(child) in seen:      # shared module reachable by two paths
                continue
            seen.add(id(child))
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            rank = None
            for pat, r in rules:
                # endswith, NOT substring. A LoRA-wrapped target exposes its
                # frozen base at "<target>.base", which CONTAINS the pattern, so
                # a substring test re-wraps it on any second injection pass --
                # and train_recovery injects twice whenever --init-adapters is
                # used (once to shape the model for the load, once for real).
                # The result was two parallel rank-r adapters per target,
                # base(x) + lora_inner(x) + lora_outer(x): effectively rank 2r
                # at 2x the adapter params. Every run before 2026-09-13 has this.
                if full.endswith(pat):
                    rank = r
                    break
            if not rank:
                continue
            setattr(parent, child_name, LoRALinear(child, rank))
            total_new += rank * (child.in_features + child.out_features)
            injected.append((full, rank))
    if verbose:
        by_rank = {}
        for name, r in injected:
            by_rank[r] = by_rank.get(r, 0) + 1
        print(f"  LoRA injected into {len(injected)} layers "
              f"({', '.join(f'{c}@r{r}' for r, c in sorted(by_rank.items()))}); "
              f"{total_new/1e6:.2f} M new params")
    return injected


def trainable_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]


def freeze_base(model, also_train=("lora_A", "lora_B", "a_lora_A", "a_lora_B",
                                   "A_log", "dt_bias")):
    """Freeze everything, then re-enable adapters and the gate parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for name, p in model.named_parameters():
        if any(k in name for k in also_train):
            p.requires_grad_(True)
            n += p.numel()
    return n
