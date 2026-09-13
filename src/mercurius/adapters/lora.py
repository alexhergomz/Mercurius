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


def merge_and_restart(model, optimizer=None):
    """ReLoRA-style rank accumulation: fold each adapter into its base, then
    restart the adapter from zero.

    A rank-r adapter can only ever move the weight within an r-dimensional
    subspace. Merging releases that constraint: after k merges the accumulated
    update can reach rank k*r while never holding more than r(m+n) trainable
    parameters at once. Measured on this model, the dense update needs rank ~300
    on in_proj_qkv, which costs 2.1 M as a single factorization but only
    r(m+n) at a time this way.

    Two details that make or break it:

    1. THE OPTIMIZER STATE MUST BE DROPPED for the restarted factors. Adam's
       moments encode the direction the old subspace was moving in; carried
       across a merge they immediately drag the freshly initialized factors back
       into the subspace we just escaped, which is the whole point of merging.

    2. THE MERGED BASE MUST BE SAVED. base.weight is frozen, so a checkpointer
       keyed on requires_grad -- which is what save_trainable does, deliberately
       -- will not write it, and every merged update is silently lost at save
       time. This is the third instance of that failure mode in this project
       (the hardcoded name filter dropped in_proj_a, the substring match
       double-wrapped adapters), so modules are tagged here and the saver reads
       the tag rather than inferring anything.
    """
    n = 0
    for mod in model.modules():
        if not isinstance(mod, LoRALinear):
            continue
        with torch.no_grad():
            delta = (mod.lora_B.float() @ mod.lora_A.float()) * mod.scale
            mod.base.weight.data += delta.to(mod.base.weight.dtype)
            nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5))
            mod.lora_B.zero_()
        mod.merged = True                      # read by save_trainable
        if optimizer is not None:
            for p in (mod.lora_A, mod.lora_B):
                optimizer.state.pop(p, None)
        n += 1
    return n


def resize_lora(model, rules, verbose=True):
    """Give each adapter the rank its rule now asks for, discarding the old one.

    Only safe AFTER merge_and_restart has folded the existing adapters into
    their bases -- otherwise the delta they carry is thrown away. That ordering
    is what lets --init-adapters (rank 16 on disk) be combined with a different
    --kda-rank: load at the rank the checkpoint has, fold it in, then resize.
    Loading a rank-16 checkpoint into rank-64 adapters fails outright, since
    strict=False forgives missing keys but not mismatched shapes.
    """
    changed = []
    for name, mod in model.named_modules():
        if not isinstance(mod, LoRALinear):
            continue
        for pat, r in rules:
            if not name.endswith(pat):
                continue
            if r and r != mod.rank:
                dev, dt = mod.lora_A.device, mod.lora_A.dtype
                mod.rank = r
                mod.scale = 1.0                      # alpha = rank convention
                mod.lora_A = nn.Parameter(torch.zeros(
                    r, mod.base.in_features, device=dev, dtype=dt))
                mod.lora_B = nn.Parameter(torch.zeros(
                    mod.base.out_features, r, device=dev, dtype=dt))
                nn.init.kaiming_uniform_(mod.lora_A, a=math.sqrt(5))
                changed.append((name, r))
            break
    if verbose and changed:
        by_r = {}
        for _, r in changed:
            by_r[r] = by_r.get(r, 0) + 1
        print(f"  resized {len(changed)} adapters "
              f"({', '.join(f'{c}@r{r}' for r, c in sorted(by_r.items()))})",
              flush=True)
    return len(changed)


def merged_base_names(model):
    """Parameter names of bases that absorbed a merge and must be checkpointed."""
    out = set()
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear) and getattr(mod, "merged", False):
            out.add(f"{name}.base.weight")
    return out


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
