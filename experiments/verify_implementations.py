"""Check that each mechanism added recently does what its docstring claims.

Every item here is a property the training runs silently depend on. None of them
show up in a loss curve when they are wrong -- which is how this project has
already shipped a checkpointer that dropped 37.7 M trained parameters, an
injector that double-wrapped every adapter, a rank flag that printed 64 and
trained 16, and a rank allocator that was never called.

Runs on CPU. No GPU, no model download.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from mercurius.adapters.lora import (LoRALinear, inject_lora, freeze_base, merge_and_restart,
                  merged_base_names, resize_lora, _lora_scale)
from mercurius.adapters.layerscale import ScaledOutput, install_layerscale
from mercurius.models.kda import fuse_gate_lora

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


class Tiny(nn.Module):
    """Two 'layers' whose names match the real rule patterns."""
    def __init__(self, dt=torch.float32):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            l = nn.Module()
            l.linear_attn = nn.Module()
            l.linear_attn.in_proj_qkv = nn.Linear(32, 96, bias=False).to(dt)
            l.linear_attn.out_proj = nn.Linear(64, 32, bias=False).to(dt)
            l.mlp = nn.Module()
            l.mlp.down_proj = nn.Linear(64, 32, bias=False).to(dt)
            self.layers.append(l)

    def forward(self, x):
        out = 0
        for l in self.layers:
            out = out + l.linear_attn.in_proj_qkv(x).sum()
        return out


RULES = [("linear_attn.in_proj_qkv", 16), ("linear_attn.out_proj", 16),
         ("mlp.down_proj", 16)]


def t_merge_is_function_preserving(dt, tol):
    """ReLoRA's merge must not change the model's output.

    delta is computed in fp32 but ADDED to a bf16 base, so the sum rounds. This
    measures how much, because an error comparable to the update itself would
    mean each merge throws away part of what was learned.
    """
    torch.manual_seed(0)
    m = Tiny(dt)
    inject_lora(m, RULES, verbose=False)
    freeze_base(m)
    for mod in m.modules():                      # give the adapters real content
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.lora_B.normal_(0, 0.02)
    x = torch.randn(4, 32, dtype=dt)
    mods = [mod for mod in m.modules() if isinstance(mod, LoRALinear)]
    before = [mod(torch.randn(4, mod.base.in_features, dtype=dt)) for mod in mods]
    xs = [torch.randn(4, mod.base.in_features, dtype=dt) for mod in mods]
    before = [mod(xi) for mod, xi in zip(mods, xs)]
    n = merge_and_restart(m)
    after = [mod(xi) for mod, xi in zip(mods, xs)]
    rel = max(float((a - b).norm() / b.norm().clamp_min(1e-30))
              for a, b in zip(before, after))
    check(f"merge preserves function ({dt})", rel < tol, f"max rel err {rel:.2e}")
    check(f"merge touched every adapter ({dt})", n == len(mods), f"{n} of {len(mods)}")
    zeroed = all(float(mod.lora_B.abs().max()) == 0.0 for mod in mods)
    check(f"lora_B restarted at zero ({dt})", zeroed)
    reinit = all(float(mod.lora_A.abs().max()) > 0.0 for mod in mods)
    check(f"lora_A reinitialised nonzero ({dt})", reinit)


def t_merged_bases_are_saveable():
    """A merged base is frozen but changed; the saver must still write it."""
    torch.manual_seed(0)
    m = Tiny()
    inject_lora(m, RULES, verbose=False)
    freeze_base(m)
    names_before = merged_base_names(m)
    merge_and_restart(m)
    names = merged_base_names(m)
    n_lora = sum(1 for mod in m.modules() if isinstance(mod, LoRALinear))
    check("no merged names before a merge", len(names_before) == 0)
    check("every merged base is named", len(names) == n_lora,
          f"{len(names)} of {n_lora}")
    sd = m.state_dict()
    check("named merged bases exist in state_dict",
          all(k in sd for k in names))
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    check("merged bases are NOT caught by requires_grad",
          not (names & trainable), "which is exactly why the saver needs them")


def t_resize_after_merge_is_exact():
    """Resizing is only safe once the adapters carry nothing."""
    torch.manual_seed(0)
    m = Tiny()
    inject_lora(m, RULES, verbose=False)
    freeze_base(m)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.lora_B.normal_(0, 0.02)
    mods = [mod for mod in m.modules() if isinstance(mod, LoRALinear)]
    xs = [torch.randn(4, mod.base.in_features) for mod in mods]
    before = [mod(xi) for mod, xi in zip(mods, xs)]
    merge_and_restart(m)
    resize_lora(m, [(p, 64) for p, _ in RULES], verbose=False)
    after = [mod(xi) for mod, xi in zip(mods, xs)]
    rel = max(float((a - b).norm() / b.norm().clamp_min(1e-30))
              for a, b in zip(before, after))
    check("merge+resize preserves function", rel < 1e-6, f"max rel err {rel:.2e}")
    ranks = {mod.rank for mod in mods}
    check("every adapter took the new rank", ranks == {64}, f"ranks {ranks}")
    shapes_ok = all(mod.lora_A.shape[0] == 64 and mod.lora_B.shape[1] == 64
                    for mod in mods)
    check("lora_A/lora_B reshaped consistently", shapes_ok)
    check("scale stays 1.0 at the new rank",
          all(abs(mod.scale - 1.0) < 1e-12 for mod in mods))


def t_resize_without_merge_loses_the_delta():
    """The ordering matters; this documents what the wrong order costs."""
    torch.manual_seed(0)
    m = Tiny()
    inject_lora(m, RULES, verbose=False)
    freeze_base(m)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.lora_B.normal_(0, 0.02)
    mods = [mod for mod in m.modules() if isinstance(mod, LoRALinear)]
    xs = [torch.randn(4, mod.base.in_features) for mod in mods]
    before = [mod(xi) for mod, xi in zip(mods, xs)]
    resize_lora(m, [(p, 64) for p, _ in RULES], verbose=False)   # no merge first
    after = [mod(xi) for mod, xi in zip(mods, xs)]
    rel = max(float((a - b).norm() / b.norm().clamp_min(1e-30))
              for a, b in zip(before, after))
    check("resize WITHOUT merge does lose the delta", rel > 1e-4,
          f"rel change {rel:.2e} -- confirms the documented ordering requirement")


def t_layerscale_identity():
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16):
        lin = nn.Linear(32, 64).to(dt)
        x = torch.randn(4, 32, dtype=dt)
        ref = lin(x)
        w = ScaledOutput(lin, 64)
        check(f"LayerScale is a bit-exact no-op at init ({dt})",
              torch.equal(ref, w(x)))
        check(f"lambda held in fp32 ({dt})", w.ls_lambda.dtype == torch.float32)


def t_layerscale_wraps_once():
    """The substring bug wrapped each target twice; endswith must not."""
    torch.manual_seed(0)
    m = Tiny()
    inject_lora(m, RULES, verbose=False)
    n, total = install_layerscale(m, targets=("mlp.down_proj",), verbose=False)
    check("LayerScale wraps each target exactly once", n == 2, f"{n} wraps for 2 layers")
    nested = sum(1 for mod in m.modules()
                 if isinstance(mod, ScaledOutput) and isinstance(mod.inner, ScaledOutput))
    check("no nested LayerScale wrappers", nested == 0)


def t_pull_to_init_direction():
    """p.lerp_(ref, k) must move p toward ref by exactly fraction k."""
    p = torch.tensor([1.0, 2.0, 3.0])
    ref = torch.tensor([0.0, 0.0, 0.0])
    d0 = float((p - ref).norm())
    p.lerp_(ref, 0.25)
    d1 = float((p - ref).norm())
    check("pull-to-init moves toward the reference", d1 < d0)
    check("pull-to-init magnitude is exact", abs(d1 / d0 - 0.75) < 1e-6,
          f"ratio {d1/d0:.6f}, expected 0.750000")
    # the arithmetic that makes the literature value inert here
    lr, strength, steps = 3e-5, 1e-3, 150
    frac = 1.0 - (1.0 - lr * strength) ** steps
    check("documented: strength 1e-3 is inert at lr 3e-5 over 150 steps",
          frac < 1e-4, f"moves {frac:.2e} of the way to init")


def t_relora_warmup_shape():
    """Reproduce the scheduler multiplier the trainer applies."""
    every, warm, total = 30, 8, 150
    last, lrs = -10**9, []
    for step in range(1, total + 1):
        lr = 1.0                              # stand-in for the annealed base LR
        since = step - last
        if 0 <= since < warm:
            lr *= max(since, 1) / warm
        lrs.append(lr)
        if every and step > 0 and step % every == 0:
            last = step
    after_merge = lrs[30:38]
    rising = all(b >= a for a, b in zip(after_merge, after_merge[1:]))
    check("LR ramps up over the warmup window", rising,
          f"{[round(v,3) for v in after_merge[:4]]} ...")
    check("LR returns to full scale after the window", abs(lrs[38] - 1.0) < 1e-12)
    check("warmup never exceeds full scale", max(lrs) <= 1.0 + 1e-12)
    check("no warmup applied before the first merge",
          all(abs(v - 1.0) < 1e-12 for v in lrs[:29]))


class FakeKDA(nn.Module):
    """Mirrors KDA's decay path: a dense projection plus an in-class adapter."""
    def __init__(self, hidden=64, out=128, rank=8, dt=torch.float32):
        super().__init__()
        self.in_proj_a = nn.Linear(hidden, out, bias=False).to(dt)
        self.lora_rank = rank
        self.a_lora_A = nn.Parameter(torch.randn(rank, hidden, dtype=dt) * 0.02)
        self.a_lora_B = nn.Parameter(torch.randn(out, rank, dtype=dt) * 0.02)

    def decay(self, x):
        a = self.in_proj_a(x)
        if self.lora_rank > 0:
            a = a + F.linear(F.linear(x, self.a_lora_A), self.a_lora_B)
        return a


def t_fuse_gate_lora():
    torch.manual_seed(0)
    holder = nn.Module()
    holder.k = FakeKDA()
    x = torch.randn(4, 64)
    before = holder.k.decay(x)
    n = fuse_gate_lora(holder, verbose=False)
    after = holder.k.decay(x)
    rel = float((before - after).norm() / before.norm())
    check("gate-lora fusion preserves the decay path", rel < 1e-6,
          f"rel err {rel:.2e}")
    check("fusion visited the layer", n == 1)
    check("lora_rank switched off so the branch is skipped",
          holder.k.lora_rank == 0)
    check("a_lora factors zeroed",
          float(holder.k.a_lora_A.abs().max()) == 0.0
          and float(holder.k.a_lora_B.abs().max()) == 0.0)
    check("a_lora factors frozen",
          not holder.k.a_lora_A.requires_grad and not holder.k.a_lora_B.requires_grad)
    # and the dense matrix genuinely absorbed it
    holder2 = nn.Module(); torch.manual_seed(0); holder2.k = FakeKDA()
    w0 = holder2.k.in_proj_a.weight.detach().clone()
    fuse_gate_lora(holder2, verbose=False)
    moved = float((holder2.k.in_proj_a.weight - w0).norm())
    check("in_proj_a absorbed the adapter", moved > 1e-6, f"moved {moved:.4f}")


def t_lora_scaling():
    """rsLoRA must fall as 1/sqrt(r); the classic convention must not move."""
    classic = [_lora_scale(r, None, False) for r in (8, 16, 32, 64)]
    check("classic scale is constant in rank", all(abs(s - 1.0) < 1e-12 for s in classic))
    rs = [_lora_scale(r, 4.0, True) for r in (8, 16, 32, 64)]
    check("rsLoRA alpha=4 matches the old scale at rank 16",
          abs(rs[1] - 1.0) < 1e-12, f"{rs[1]:.4f}")
    ratios = [rs[i] / rs[i + 1] for i in range(3)]
    check("rsLoRA halves every 4x rank (sqrt law)",
          all(abs(x - 2 ** 0.5) < 1e-9 for x in ratios),
          f"ratios {[round(x,4) for x in ratios]}")
    check("rsLoRA alpha=8 gives the common scale of 2 at rank 16",
          abs(_lora_scale(16, 8.0, True) - 2.0) < 1e-12)
    # resize must carry the convention, not reset to 1.0
    torch.manual_seed(0)
    m = Tiny()
    inject_lora(m, RULES, verbose=False, alpha=4.0, rslora=True)
    freeze_base(m)
    merge_and_restart(m)
    resize_lora(m, [(p, 64) for p, _ in RULES], verbose=False)
    got = {round(mod.scale, 6) for mod in m.modules() if isinstance(mod, LoRALinear)}
    check("resize recomputes scale under rsLoRA, not back to 1.0",
          got == {round(4.0 / 8.0, 6)}, f"scales {got}, expected 0.5 at rank 64")


def main():
    print("verifying recently added mechanisms\n")
    t_merge_is_function_preserving(torch.float32, 1e-6)
    t_merge_is_function_preserving(torch.bfloat16, 5e-2)
    print()
    t_merged_bases_are_saveable()
    print()
    t_resize_after_merge_is_exact()
    t_resize_without_merge_loses_the_delta()
    print()
    t_layerscale_identity()
    t_layerscale_wraps_once()
    print()
    t_pull_to_init_direction()
    print()
    t_relora_warmup_shape()
    print()
    t_fuse_gate_lora()
    print()
    t_lora_scaling()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
